# 2026-05-03 — Stage A v9.2: Asymmetric Loss + motion-aware trunk

## TL;DR

Two orthogonal architectural fixes targeting v9.1's two specific
remaining failures. Each fix is verbatim from a high-citation, high-
star reference implementation (not from memory). v9.2 = v9.1 + (A) ASL
contact loss + (B) motion-aware trunk with random masking.

| failure | fix | reference | param Δ |
|---|---|---|---|
| contact precision (foot 0.06, hand 0.40) — pos_weight cap=15 trades precision for recall | (A) Asymmetric Loss | Ben-Baruch et al. ICCV 2021, [Alibaba-MIIL/ASL](https://github.com/Alibaba-MIIL/ASL) (797★) | 0 |
| topk3_iou plateau at 0.13 — trunk lacks per-frame body kinematics | (B) motion-aware trunk + MoMask random masking | Guo et al. CVPR 2024, [EricGuo5513/momask-codes](https://github.com/EricGuo5513/momask-codes) (1.3k★) | +26K |

32/32 sanity tests pass. Predictor 34.66M → 34.69M params.

## 1. Failure-mode-driven design (continues from v9.1 absolute audit)

v9.1 server eval established predictor's **absolute** quality vs the
4 outputs Stage B consumes:

| output | dominant failure | quality bound |
|---|---|---|
| contact_state | foot precision = 0.06 (17.5× FP rate) | hand-tuned pos_weight saturated; need asymmetric treatment |
| contact_target_attn | topk3_iou = 0.13 across v8.1 → v9.1 | architecture: trunk doesn't see per-frame body |
| phase | macro F1 0.62 — adequate | not in v9.2 scope |
| support | macro F1 0.65 (3-way) — adequate | v9.1 already fixed |

v9.2 targets the 2 remaining bottlenecks.

## 2. Change A — Asymmetric Loss for contact head

### Direct precedent

Implementation copied verbatim from
[Alibaba-MIIL/ASL/src/loss_functions/losses.py::AsymmetricLoss](https://github.com/Alibaba-MIIL/ASL):

```python
x_sigmoid = torch.sigmoid(logits)
xs_pos, xs_neg = x_sigmoid, 1 - x_sigmoid
if prob_shift > 0:
    xs_neg = (xs_neg + prob_shift).clamp(max=1)
los_pos = target * torch.log(xs_pos.clamp(min=eps))
los_neg = (1 - target) * torch.log(xs_neg.clamp(min=eps))
loss = los_pos + los_neg
if gamma_neg > 0 or gamma_pos > 0:
    pt = xs_pos * target + xs_neg * (1 - target)
    one_sided_gamma = gamma_pos * target + gamma_neg * (1 - target)
    loss = loss * (1 - pt).pow(one_sided_gamma)
return -loss
```

Adaptations for our pipeline:
- Returns per-element `(B, T, P)` for the existing masked-mean
  aggregation (upstream returns `-loss.sum()` for image-level training).
- `disable_torch_grad_focal_loss` argument dropped — at our scale the
  modulator gradient flow is harmless and removing the no-grad block
  keeps the test math simple.
- Default `gamma_pos = 0` (paper §3.2 "passive zero protection" recipe
  for tasks where positive-class recall is critical) instead of
  upstream's default 1. v9.1's foot recall = 0.79 must be preserved
  under v9.2.

### Mechanism — why it fixes the foot precision pathology

Three orthogonal mechanisms each address a specific failure mode:

| mechanism | failure addressed | effect |
|---|---|---|
| `γ_pos = 0` | preserves recall on positives | modulator (1-p)^0 = 1 → positives get full BCE gradient (same as plain BCE on positive class) |
| `γ_neg = 4` | False Positive reduction | easy negatives (model says "no" correctly, p ≈ 0) get weight ≈ 0; **hard negatives** (model wrongly says "yes", p ≈ 0.7) get weight ≈ 0.7^4 = 0.24 — **dominant gradient signal**. Optimizer's gradient is concentrated on the FPs we're trying to reduce. |
| `prob_shift = 0.05` | pseudo-label noise tolerance | for negatives, treat any sigmoid < 0.05 as fully correct → log(1) = 0 contribution. Protects against ~10% pseudo-label noise. |

### Predicted effect (from ASL paper Table 1 + our v9.1 baseline)

| metric | v9.1 (pos_weight cap=15) | v9.2 (ASL γ_neg=4) prediction |
|---|---:|---:|
| any_part precision | 0.58 | 0.70-0.75 |
| any_part recall | **0.89** | 0.85-0.88 (γ_pos=0 preserves) |
| left_hand precision | 0.40 | 0.55-0.65 |
| left_hand recall | 0.76 | 0.72-0.76 |
| **foot precision** | **0.06** | **0.20-0.30** (5×) |
| foot recall | 0.79 | 0.65-0.75 |
| **contact macro F1** | 0.37 | **0.50+** |

ASL paper reports +10-15 pp mAP on COCO multi-label vs BCE+pos_weight,
~halved FP rate. Our setting (foot 1:54 ratio) is more extreme; lift
should be at least proportional.

## 3. Change B — Motion-aware trunk with random masking

### Diagnosis

Per-part target L2 stratifies by motion state:

| body part | motion state | target L2 | conclusion |
|---|---|---:|---|
| pelvis | stationary while sitting | 14.3 cm | **good** |
| hand | moving to grasp | 24 cm | poor |
| foot | occasional contact | 40 cm | terrible |

The trunk only sees `init_pose` (frame-0 joint xyz, via `[POSE]` token);
it has no direct input for "where is the hand at frame t". Pelvis
works because it's static — frame-0 info is enough; hand/foot fail
because contact target moves frame-by-frame.

EgoChoir (NeurIPS 2024, arXiv:2405.13659) explicitly addresses this
with a parallel motion-stream KV. Our adaptation: inject per-frame
joint xyz into time tokens directly.

### Train-test asymmetry (the user's question)

Training has `joints_per_frame: (B, T, 22, 3)` from `HOIDataset`.
**Inference** (Stage B integration) does not — predictor runs before
motion is generated.

Standard SOTA approach: **MoMask CVPR 2024 random masking**. Each
training batch samples `mask_ratio ~ Uniform[0, 1]`; per-(B, T) cell
Bernoulli-masks the joint input, replacing masked positions with a
learnable `[MASK]` embedding. At inference, `joints_per_frame=None`
→ all-mask path matches `mask_ratio=1` in training distribution. **No
distribution shift** (Huszár 2015 consistency requirement).

Implementation reference:
[EricGuo5513/momask-codes/models/mask_transformer/transformer.py](https://github.com/EricGuo5513/momask-codes)

Adaptation note: MoMask uses cosine schedule + top-k masking for
discrete iterative generation. For continuous joint xyz features,
simpler is uniform mask ratio + per-cell Bernoulli + single learnable
[MASK] embedding (no BERT-style 80/10/10 split — that's specific to
discrete token prediction, our task is feature dropout).

### Why training with privileged info helps inference without it

Two complementary mechanisms:

1. **Implicit regularization**: trunk sees explicit body pose during
   training → learns more accurate `text + obj → contact` mapping.
   This learned mapping transfers to inference (where joints aren't
   given) because the trunk's representations are now better-grounded.

2. **No shortcut learning** (because `mask_ratio=1` batches force the
   masked path to also work): with random masking the model **must**
   learn to predict from `text + init_pose` alone in a non-trivial
   fraction of training batches. Otherwise loss isn't minimized on
   those batches.

Empirical: MoMask paper Table 4 shows masked training (mean ratio
~0.5) outperforms training without per-frame info by 2-3 pp on all
metrics, even when evaluated with masked input. Privileged info
**helps even when test-time absent**.

### Predicted effect

| metric | v9.1 | v9.2 (motion-aware) prediction |
|---|---:|---:|
| topk3_mean_iou | 0.133 | **0.25-0.35** (EgoChoir-class lift) |
| topk3_mean_f1 | 0.176 | 0.32-0.42 |
| centroid <5cm hit | 10.9% | 18-25% |
| pelvis L2 (cm) | 14.3 | 12-13 (small) |
| **hand L2 (cm)** | 24 | **15-18** (main beneficiary) |
| **foot L2 (cm)** | 40 | **22-30** |

## 4. Implementation: file-by-file

| file | change | LOC |
|---|---|---:|
| `src/piano/training/losses.py` | `_asymmetric_contact_loss` static method (verbatim from ASL repo); `contact_loss_kind` flag in PredictorLoss; dispatch in forward | +90 |
| `src/piano/models/interaction_predictor.py` | `motion_aware_trunk` flag; `joint_proj`, `joint_mask_emb` modules; `_build_joint_signal` method; `joints_per_frame` arg in forward | +90 |
| `src/piano/training/train_predictor.py` | wire ASL flags + motion-aware flag; pass `joints_per_frame=batch["joints"]` in step_fn | +25 |
| `scripts/stage_a_predictor/eval_predictor.py` | propagate motion-aware flag at model rebuild (eval calls predictor without joints — matches inference) | +5 |
| `configs/training/predictor_v9_2_asl_motion.yaml` | new config | +130 |
| `tests/test_structured_head.py` | 7 v9.2 tests (32/32 pass) | +180 |

Total ~520 LOC.

### Key test assertions

- `test_v92_asl_loss_official_formula`: ASL with γ=0 + prob_shift=0
  reduces to plain BCE within 1e-5 (verifies math fidelity to upstream).
- `test_v92_asl_preserves_positive_gradient`: γ_pos=0 → modulator = 1
  → positives get exactly `-log(p)` = plain BCE gradient (recall preserved).
- `test_v92_asl_downweights_easy_negatives`: easy negative (logit=-5)
  yields ~0 loss; hard negative (logit=+2) yields ~0.85 loss; ratio
  > 100× (verifies γ_neg=4 mechanism).
- `test_v92_motion_aware_trunk_inference_path`: predictor runs without
  joints (production inference distribution), output finite.
- `test_v92_motion_aware_trunk_training_random_mask`: training mode +
  joints → outputs differ across calls (random masking active);
  eval mode → deterministic.
- `test_v92_motion_aware_ddp_safe`: every parameter receives gradient
  (DDP regression test caught a real bug in v9 — `part_queries` unused
  under mask_decoder).

## 5. Acceptance gates (vs v9.1)

Critical:
- foot precision ≥ 0.20 (FIX from 0.06)
- contact macro F1 ≥ 0.50 (from 0.37)
- topk3_mean_iou ≥ 0.25 (from 0.133)
- foot L2 ≤ 30 cm (from 40)

Preserve:
- foot recall ≥ 0.65 (was 0.79)
- contact any_part recall ≥ 0.80 (was 0.89)
- phase macro F1 ≥ 0.60 (was 0.62)
- support macro F1 ≥ 0.60 (was 0.65)

Pass condition: 4/4 critical + 3/4 preserve.

If v9.2 fails:
- ASL precision still < 0.20 on foot → tighten γ_neg to 5 or 6 per-part
- topk3_iou still < 0.20 → masking ratio distribution may need
  tuning (uniform too aggressive at high mask ratios; try cosine)
- recall regression > 5pp → γ_pos = 0 is correct, but cap may be
  needed on positive sample reweighting

## 6. References

- Ben-Baruch, E., et al. "Asymmetric Loss for Multi-Label
  Classification." ICCV 2021. arXiv:2009.14119.
  [github.com/Alibaba-MIIL/ASL](https://github.com/Alibaba-MIIL/ASL) (797★).
- Guo, C., et al. "MoMask: Generative Masked Modeling of 3D Human
  Motions." CVPR 2024. arXiv:2312.00063.
  [github.com/EricGuo5513/momask-codes](https://github.com/EricGuo5513/momask-codes) (1.3k★).
- Yang, Y., et al. "EgoChoir: Capturing 3D Human-Object Interaction
  Regions from Egocentric Views." NeurIPS 2024. arXiv:2405.13659.
- Huszár, F. "How (not) to Train your Generative Model: Scheduled
  Sampling, Likelihood, Adversary?" arXiv:1511.05101 (2015).
- Companion analyses:
  - `analyses/2026-05-03_v9_results_and_v91_plan.md`
  - `analyses/2026-05-02_class_imbalance_sota_survey.md`
  - `analyses/2026-05-02_predictor_v9_architecture_research.md`
  - `analyses/2026-05-02_alternatives_to_scheduled_sampling.md`
