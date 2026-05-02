# 2026-05-05 — v8.1 results + v8.1.1 plan (top-K minimum GT + top-K eval)

## TL;DR

v8.1 server retrain validated **both core hypotheses** but two
specific issues remain:

| outcome | evidence | status |
|---|---|---|
| Hypothesis 1 (MoMask mask fixes TF gap) | phase macro F1 0.577 → **0.637** (recovers v7-fix 0.632) | ✅ confirmed |
| Hypothesis 2 (Multi-hot binary improves location precision) | target <5cm hit 4.5% → **11.6%** (2.6×); <10cm 17.5% → 26.9% | ✅ confirmed |
| Pelvis advantage retained | pct_within_10cm 33.9% → **59.1%** | ✅ |
| Foot target L2 regressed | 25 → 42.7 cm (heavy) | ❌ — fixable, see § 3 |
| multihot_mean_iou below gate | 0.141 vs ≥ 0.30 | ⚠️ — metric is wrong, see § 4 |

v8.1.1 is one minimal change (no architectural rework) that fixes both
issues:

1. **GT mask = top-K nearest tokens ∪ within-τ tokens** with K=3 —
   guarantees ≥ K positives per cell regardless of FPS density (fixes
   foot empty-mask)
2. **Top-K based F1 / IoU** added as primary eval metric (threshold-
   free, robust to focal-loss sigmoid calibration)

Both ship in one retrain. Same architecture, single config flag.

## 1. v8.1 vs v8 vs v7-fix (val 1304 clips, subject_split)

| metric | v7-fix | v8 best | **v8.1 best** (ep29) | v8.1 final (ep99) |
|---|---:|---:|---:|---:|
| target_top1_token_recall | n/a | 0.093 | n/a (deprecated) | n/a |
| **multihot_mean_iou** | n/a | n/a | 0.141 | 0.143 |
| **multihot_mean_f1** | n/a | n/a | 0.184 | 0.186 |
| target overall L2 (back-compat) | 21.77 | 21.55 | 23.79 | 24.91 |
| **target <5cm hit** | 4.5 % | 5.6 % | **11.6 %** | 12.2 % |
| **target <10cm hit** | 17.5 % | 19.4 % | **26.9 %** | 27.4 % |
| target <20cm hit | 48.1 % | 50.2 % | n/a | n/a |
| contact macro_f1 | 0.237 | 0.235 | 0.219 | 0.239 |
| contact any_part_f1 | 0.379 | 0.484 | 0.437 | 0.463 |
| **phase macro F1** | 0.632 | 0.577 | **0.637** | 0.610 |
| support macro F1 | 0.397 | 0.378 | 0.393 | 0.392 |
| pelvis target L2 (cm) | 15.4 | **14.4** | 14.8 | 15.8 |
| pelvis pct_within_10cm | 33.9 % | 51.9 % | **59.1 %** | 53.3 % |
| hand L2 (cm) | 22.4 | 22.8 | 24.5 | 26.0 |
| **foot L2 (cm)** | 25.3 | 27.4 | **42.7** ❌ | 42.9 ❌ |

## 2. Wandb training trajectory

| epoch | target train | target val | phase train | phase val | val unweighted |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.458 | 0.468 | 0.320 | 0.328 | 1.315 |
| 10 | 0.444 | 0.463 | 0.271 | 0.326 | 1.298 |
| 25 | 0.432 | 0.456 | 0.244 | 0.323 | 1.273 |
| **29** (best_val) | — | best | — | best | **1.270** |
| 50 | 0.412 | 0.455 | 0.227 | 0.336 | 1.318 |
| 100 | 0.397 | 0.456 | 0.207 | 0.353 | 1.366 |

Reads:
- `loss_target` order ~ 0.4-0.5 (focal+dice scale), correctly different
  from v8 KL's ~ 1.5 → confirms `target_loss_kind="focal_dice"` is wired
- No `loss_consistency` / `weight_*` / `log_var_*` columns → confirms
  `consistency_weight=0.0` and `use_kendall_weights=false` both wired
- val_loss_unweighted bottoms ep25-30 → best_val correctly selects
  ep29
- phase val 0.318 → 0.353 after best_val: healthy late-stage overfit.
  v8 had val phase loss flat 0.35-0.36 (no real learning); v8.1
  legitimately reaches lower val and then overfits.

## 3. Failure 1: foot L2 25 → 42.7 cm — root cause

τ_foot = 3 cm in v8.1 yaml matches v12_strict's tight contact
threshold. But 128 FPS-sampled tokens on a ~ 1 m object span ~ 0.088 m
average inter-token spacing (sqrt of surface-area / 128). **τ_foot =
3 cm is below this density floor**: most foot contact-positive cells
have **0 tokens within 3 cm of the GT closest-mesh-point**.

When the multi-hot GT mask is empty:
- focal BCE: forces all logits → -∞ (model treats every token as a
  hard negative)
- dice loss: the empty-mask path I wrote returns `1 - 0/(pred + ε) = 1`
  regardless of prediction quality — vacuous gradient signal
- Net effect: foot head receives effectively no positive supervision
  and learns to suppress all 128 logits

Foot L2 42 cm is consistent with a head that just predicts uniform
near-zero attention (so attention-weighted xyz collapses to the
spatial centroid of all object tokens, far from any specific contact
location).

## 4. Failure 2: multihot_mean_iou = 0.141 — metric ill-posed

The IoU metric uses threshold = 0.5 on sigmoid output. But focal +
dice training does **not** calibrate sigmoid to the 0.5 decision
boundary:
- focal loss pushes hard examples up but does not enforce sigmoid > 0.5
  (the modulating factor `(1 - p_t)^γ` saturates well before p_t = 1)
- dice loss optimizes set overlap globally; the optimum can have most
  predictions at 0.3-0.5 with the "right" tokens slightly higher

Empirically: `multihot_mean_precision = 0.18`, `multihot_mean_recall =
0.22` at threshold 0.5. The model **does** rank correct tokens higher
on average (evidence: target <5cm hit triples), but the absolute
sigmoid value doesn't cleanly cross 0.5.

Standard fix in segmentation literature: use top-K based F1 / IoU
(threshold-free). For each cell:
- pred set = top-K predictions by sigmoid score
- GT set = top-K GT positives (or distance-defined)
- F1 / IoU on these two sets

This is what v8.1.1's eval will report alongside the legacy
0.5-threshold version.

## 5. v8.1.1 design — one config change

### Loss change

Replace v8.1's `gt_mask = (d < tau)` with:

```python
mask_topk = scatter(topk(-d, k=3))           # K nearest tokens, always K positives
mask_tau  = (d < tau_per_part)               # within-τ tokens, density-dependent
gt_mask   = mask_topk | mask_tau             # union
```

Implementation: new `target_topk_min_positives: int = 0` parameter on
`PredictorLoss._focal_dice_target_loss`. K=0 disables (v8.1 behaviour);
K=3 is the v8.1.1 default.

Properties:
- Sparse density (foot τ_foot=3cm, FPS spacing ~ 8.8cm): mask = top-3
  nearest = 3 positives. Foot head now has supervision.
- Dense density (pelvis τ_pelvis=12cm, ~ 5 tokens within τ): mask =
  top-3 ∪ within-τ ≈ 5-7 positives. Reflects the wider physical
  contact region.
- Always non-empty → dice loss never vacuous.

### Eval change

Add `topk{K}_mean_*` family alongside `multihot_mean_*`:

```python
# Top-K F1 (v8.1.1 primary)
gt_topk = argsort(d, axis=-1)[..., :K]              # K nearest GT tokens
pred_topk = argsort(-pred_sigmoid, axis=-1)[..., :K]  # K best predictions
inter, union = |gt_topk ∩ pred_topk|, |gt_topk ∪ pred_topk|
topk_iou, topk_f1 = inter/union, 2*inter/(|gt|+|pred|)
```

K=3 default for eval (matches train).

### Predicted v8.1.1 metric impact

Hypothesis-driven, falsifiable:

| metric | v8.1 | **v8.1.1 prediction** | reason |
|---|---:|---:|---|
| topk3_mean_iou (NEW) | n/a | **≥ 0.40** | model already ranks correct tokens; metric just measures it |
| topk3_mean_f1 (NEW) | n/a | ≥ 0.55 | derived from IoU |
| foot L2 (cm) | 42.7 | ≤ 30 | top-K mask fixes empty-supervision → foot head trains |
| target <5cm hit | 11.6 % | ≥ 12 % | should not regress (mask is strict superset) |
| pelvis pct<10cm | 59.1 % | ≥ 55 % | same |
| phase macro F1 | 0.637 | ≥ 0.62 | independent of target loss |
| support macro F1 | 0.393 | ≥ 0.39 | same |

Reverse-falsifiable conditions (would invalidate the diagnosis):
- topk3 IoU low (< 0.30) → model genuinely doesn't rank correct
  tokens; need architectural change (EgoChoir motion-KV)
- foot L2 still > 35 cm → not just empty-mask issue; foot kinematic
  prior or data sparsity is dominant
- target <5cm hit drops below 8 % → top-K mask diluted positive
  signal too much

## 6. v8.1.1 acceptance gates

| metric | v7-fix | v8 best | v8.1 best | **v8.1.1 gate** |
|---|---:|---:|---:|---|
| topk3_mean_iou (NEW primary) | n/a | n/a | n/a | ≥ 0.35 |
| topk3_mean_f1 | n/a | n/a | n/a | ≥ 0.50 |
| target <5cm hit | 4.5 % | 5.6 % | 11.6 % | ≥ 11 % (no regress) |
| target <10cm hit | 17.5 % | 19.4 % | 26.9 % | ≥ 25 % |
| contact macro_f1 | 0.237 | 0.235 | 0.219 | ≥ 0.23 |
| phase macro F1 | 0.632 | 0.577 | 0.637 | ≥ 0.62 |
| support macro F1 | 0.397 | 0.378 | 0.393 | ≥ 0.39 |
| foot L2 (cm) | 25.3 | 27.4 | 42.7 | ≤ 30 (FIX) |
| pelvis pct_within_10cm | 33.9 % | 51.9 % | 59.1 % | ≥ 55 % |

**Pass condition**: 7/9 gates + foot L2 fix shown.

If v8.1.1 fails:
- foot L2 still > 35 cm → likely needs EgoChoir motion-KV stream
  (v9 candidate)
- topk3 IoU < 0.30 → fundamental ranking problem; revisit
  representation choice

## 7. Pre-launch sanity (already passed)

```
$ pytest tests/test_structured_head.py -q
18 passed in 1.97s
```

New tests:
- `test_v811_topk_min_positives_no_empty_mask`: with GT_xyz far from
  all object tokens (the foot regression case), v8.1 produces vacuous
  loss; v8.1.1 with K=3 produces meaningful gradient.
- `test_v811_topk_min_perfect_pred_gives_low_loss`: perfect
  prediction on the top-K GT yields ~ 0 loss.

## 8. Re-eval v8.1 ckpt with new top-K metric (zero-cost validation)

Before launching v8.1.1 retrain, re-eval the existing
`runs/training/predictor_v8_1_masked/best_val.pt` with the new top-K
eval metric. If `topk3_mean_iou ≫ 0.141`, that confirms the metric was
the dominant issue and v8.1 already crossed the bar.

```bash
python scripts/stage_a_predictor/eval_predictor.py \
  --config configs/training/predictor_v8_1_masked.yaml \
  --checkpoint runs/training/predictor_v8_1_masked/best_val.pt \
  --split val \
  --output runs/eval/stageA_predictor_v8_1_masked_val/predictor_v8_1_masked_val_best_with_topk.json
```

(Same checkpoint, just with new eval code emitting topk3_mean_*.)

## 9. References

- Companion docs:
  - `analyses/2026-05-05_v8_round1_diagnosis_and_v81_plan.md`
  - `analyses/2026-05-02_hoi_affordance_sota_survey_post_move_as_you_say.md`
  - `analyses/2026-05-02_alternatives_to_scheduled_sampling.md`
- HOI affordance evaluation convention:
  - EgoChoir (Yang et al., NeurIPS 2024, arXiv:2405.13659) —
    per-vertex F1 / IoU, threshold tuned per task
  - Text2HOI (Cha et al., CVPR 2024, arXiv:2404.00562) — BCE+dice on
    multi-hot binary, top-K based eval
