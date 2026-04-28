# PIANO Plan

Compact action plan as of 2026-04-29.

## Immediate Priority

Stage B is stuck around `32 cm` full contact on the matched 80-clip eval, while
GT roundtrip is `18.47 cm`. Do not launch another blind decoded-contact weight
sweep. The next step is to identify where the gap enters the pipeline.

Current diagnosis: `z_int` is active, but the full soft-RVQ decoded contact
surrogate is not closing the discrete sample-time contact gap. This matches
recent HOI/control literature: OMOMO uses hand positions as an intermediate
representation, InterDiff inserts interaction correction during denoising,
CHOIS applies contact guidance during sampling, and MaskControl uses
training-time differentiable sampling plus inference-time logits/codebook
optimization. For PIANO, start with no-retrain oracles before changing
training.

### 1. K-sample oracle

Goal: test whether the current generator distribution already contains
good-contact samples.

Input checkpoint identifier on the server:

```text
runs/training/generator_v12_decoded_contact_w02_diagnostics/best_val.pt
```

Suggested measurement:

- sample K variants per eval clip with fixed prompts/object conditions;
- score each variant using the existing contact-distance metric;
- report mean, median, best-of-K, and per-subset best-of-K;
- keep the same 80-clip stratified set used by v0.12.

Runner:

```bash
python scripts/stage_b_generator/k_sample_oracle.py \
  --config runs/sweeps/stageB_v12_decoded_contact_weight_sweep/configs/generator_v12_decoded_contact_w02_diagnostics.yaml \
  --ckpt runs/training/generator_v12_decoded_contact_w02_diagnostics/best_val.pt \
  --output-dir runs/eval/stageB_v0_12_w02_bv_k16_oracle \
  --num-clips-per-subset 20 \
  --k 16 \
  --save-best
```

Decision:

- If oracle best approaches GT roundtrip, use reranking/guidance.
- If oracle best remains near 32 cm, the model distribution itself is wrong.

### 2. RVQ mixed oracle

### 2. Soft-hard gap diagnostic

Goal: directly measure whether C2b's soft decoded path is optimistic relative
to the hard generated RVQ path.

Compare on the same fixed 80 clips:

- soft full-RVQ decoded contact from the training auxiliary path;
- argmax base + residual rollout;
- sampled base + residual rollout;
- ordinary eval contact.

Decision:

- soft good but hard bad: use hard/ST-Gumbel consistency or inference-time
  logits/embedding optimization.
- soft and hard both bad: change the learned distribution or contact
  representation, not only the relaxation.

### 3. RVQ mixed oracle

Goal: locate whether base tokens or residual RVQ tokens dominate the gap.

Compare:

- GT all-RVQ decode.
- predicted all-RVQ decode.
- GT base + predicted residual.
- predicted base + GT residual.
- optionally predicted base + teacher-forced early residual layers.

Decision:

- GT base + predicted residual bad: residual/full-RVQ prediction is a major
  bottleneck.
- predicted base + GT residual bad: base plan is already wrong.
- both mixed paths good but predicted all-RVQ bad: autoregressive coupling or
  sampling instability is the bottleneck.

### 4. Subset-specific codebook audit

Goal: explain why 80-clip GT roundtrip has a nontrivial gap, especially IMHD.

Current v12 decomposition:

| subset | GT orig | GT roundtrip | full |
|---|---:|---:|---:|
| chairs | 12.04 | 12.09 | 19.07 |
| imhd | 8.41 | 22.55 | 42.30 |
| neuraldome | 14.80 | 19.30 | 33.63 |
| omomo_correct_v2 | 17.11 | 19.95 | 32.27 |

Decision:

- If failures concentrate in codebook roundtrip, representation/codebook work
  may be justified for that subset.
- If roundtrip is fine but prediction is poor, keep focus on generator.

## After Diagnostics

Only choose the next training direction after the diagnostics above.

Possible branches:

- Distribution contains good samples: add contact-aware reranking, improve
  inference guidance, or train a lightweight scorer.
- Soft-hard gap is large: move decoded-contact supervision closer to hard
  sampling via ST-Gumbel/DES-style training, and revisit MaskControl-style
  logits or embedding optimization.
- Base-token bottleneck: revise base MaskTransformer supervision/conditioning.
- Residual bottleneck: strengthen residual `z_int` path or residual training
  objective.
- Codebook bottleneck: subset-specific VQ audit, then consider representation
  fixes instead of generator-only training.

## Do Not Prioritize Now

These have evidence against them:

- More CE-only training.
- More decoded-contact weight sweeps.
- Mirror augmentation.
- Repeating trainable-copy InterControl without a new diagnostic reason.
- Base-logit-only B3 guidance as the main solution.
- Checkpoint-selection tuning as the main strategy.
- Stage C joint finetune before Stage B bottleneck diagnosis.

## Stable Later Work

Stage C remains conceptually useful: joint finetune with consistency between
predicted `z_int`, generated motion, and re-extracted interaction signals. It
should wait until Stage B can generate contact in the right band.

Paper framing should use:

- object-adaptive interaction strategy as the core claim;
- structured `z_int` as the interpretable middle layer;
- contact-distance and object-adaptive swap/text-only tests as the key evidence.
