# PIANO Plan

Compact action plan as of 2026-04-29.

## Immediate Priority

Stage B single-sample generation is stuck around `32 cm` full contact on the
matched 80-clip eval, while GT VQ roundtrip is `18.47 cm`. The K-sample oracle
shows the current generator distribution contains close-to-object samples:
K=16 best-of-K reaches `17.93 cm` mean contact, with `70%` of clips under
`22 cm` and `80%` under `25 cm`.

Current diagnosis: `z_int` is active, and the learned distribution is not
fundamentally blind to object position. However, visual review shows
distance-only reranking is not enough: the body is near the object, but its
action timing often does not bind to the object's motion. The next problem is
sample-time temporal coupling/guidance, not just spatial proximity. This
matches recent HOI/control literature: OMOMO uses hand positions as an
intermediate representation, InterDiff inserts interaction correction during
denoising, CHOIS applies contact guidance during sampling, and MaskControl uses
training-time differentiable sampling plus inference-time logits/codebook
optimization.

### 1. K-sample oracle

Result: succeeded on the v12 w02 best-val checkpoint.

Checkpoint identifier on the server:

```text
runs/training/generator_v12_decoded_contact_w02_diagnostics/best_val.pt
```

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

- K=16 best-of-K mean: `17.93 cm`, essentially the GT roundtrip band.
- Single sample mean: `32.22 cm`, matching the previous v12 contact band.
- Saved best samples re-measured at `18.70 cm` by `measure_contact_distance.py`.
- Visual review: distance-reranked samples are close to objects but often not
  temporally bound to object motion.
- Next branch: composite reranking/guidance using both spatial distance and
  moving-object kinematic coupling.

### 2. Temporal coupling diagnostic

Goal: test whether low-distance samples actually move with the object.

Runner:

```bash
python scripts/stage_b_generator/measure_temporal_coupling.py \
  --input-dir runs/eval/stageB_v0_12_w02_bv_k16_oracle/best \
  --output-dir runs/eval/stageB_v0_12_w02_bv_k16_oracle/temporal_coupling
```

Result on K=16 distance-reranked best:

| metric | value |
|---|---:|
| ordinary mean contact distance | 0.187 m |
| moving frames with any close tracked body part | 0.475 |
| moving frames with kinematic coupling | 0.323 |
| moving frames close but uncoupled | 0.245 |

Decision: distance-only reranking is insufficient as the final baseline. The
next no-retrain baseline should rerank samples by a composite of contact
distance and kinematic coupling.

### 3. Soft-hard gap diagnostic

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

### 4. RVQ mixed oracle

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

### 5. Subset-specific codebook audit

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

## After K-Sample Oracle

Immediate branch:

- Implement composite best-of-K reranking: distance + moving-object kinematic
  coupling.
- Compare distance-only K=16 vs composite K=16 on the same 80 clips.
- Prioritize IMHD and NeuralDome hard cases, where temporal coupling is weakest.

Composite rerank runner:

```bash
python scripts/stage_b_generator/k_sample_oracle.py \
  --config runs/sweeps/stageB_v12_decoded_contact_weight_sweep/configs/generator_v12_decoded_contact_w02_diagnostics.yaml \
  --ckpt runs/training/generator_v12_decoded_contact_w02_diagnostics/best_val.pt \
  --output-dir runs/eval/stageB_v0_12_w02_bv_k16_composite_oracle \
  --num-clips-per-subset 20 \
  --k 16 \
  --selection-metric composite \
  --coupling-weight 0.12 \
  --uncoupled-penalty 0.05 \
  --save-best
```

Secondary diagnostics, only if reranked samples fail visually or hard subsets
remain unacceptable:

- Soft-hard gap: measure whether the decoded training path is optimistic
  relative to hard generation.
- RVQ mixed oracle: locate whether base tokens or residual RVQ tokens dominate
  remaining outliers.
- Codebook bottleneck: subset-specific VQ audit, especially IMHD.

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
