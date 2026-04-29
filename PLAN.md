# PIANO Plan

Compact action plan as of 2026-04-30.

## Immediate Priority

Stage B single-sample generation is no longer completely stuck at the old
`31-32 cm` contact band: v14 sampled-ST `best_contact` reaches `27.37 cm` full
contact on the matched 80-clip eval. This is a real spatial-contact gain, but
GT VQ roundtrip is still `18.47 cm`, and temporal coupling barely moves
(`0.2765` moving-coupled frame fraction vs v13 `0.2653` and v12 `0.2639`).

Current diagnosis: `z_int` is active, and v14 proves that making the decoded
auxiliary path harder and closer to sampling improves both one-shot spatial
contact and the K-sample candidate pool. However, ordinary one-shot sampling
still often fails to bind to the object's motion. v14 K=16 composite reaches
`17.94 cm` after remeasure with moving-coupled `0.3715`, while v14 one-shot is
`27.37 cm` / `0.2765`, so the next lever is selection or sample-time guidance.

Latest evaluated branch: v14 keeps v13's object-local `contact_target_xyz`
trajectory loss, but takes decoded-aux logits from the all-mask MaskGIT/CFG
first step and decodes with straight-through Gumbel hard codebook samples
through the full residual RVQ path.

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

Composite K=16 result:

| metric | distance K=16 | composite K=16 |
|---|---:|---:|
| contact mean | 17.93 cm | 18.08 cm |
| moving coupled frame frac | 0.323 | 0.351 |
| close but uncoupled moving frac | 0.245 | 0.222 |

Only `12/80` selections changed. Offline rescoring shows even a max-coupled
oracle over the stored K=16 candidates reaches only about `0.390` moving
coupled frame fraction, at `20.67 cm` contact. This means the K=16 pool itself
does not contain enough strongly coupled samples. IMHD is the main blocker:
only `2/20` moving IMHD clips have any K=16 candidate with coupling >= `0.5`.

### 3. v13/v14 outcomes

v13 target trajectory improved the soft decoded auxiliary path but not hard
sampling:

| run | contact | moving close | moving coupled | close but uncoupled |
|---|---:|---:|---:|---:|
| v12 w02 best_val | 31.82 cm | 0.296 | 0.264 | 0.140 |
| v13 best_val | 31.57 cm | 0.334 | 0.265 | 0.171 |
| v14 best_contact | 27.37 cm | 0.343 | 0.277 | 0.172 |
| v12 K=16 composite | 18.08 cm | 0.473 | 0.351 | 0.222 |
| GT roundtrip | 18.47 cm | - | - | - |

v13 RVQ diagnostics showed a large gap between `soft_train_full`
(`14.78 cm`, moving coupled `0.443`) and sampled/mixed prediction paths
(`mixed_pred_all` `33.50 cm`, `mixed_pred_base_gt_residual` `35.92 cm`).
v14 directly targets this gap and gives a real single-sample contact gain, but
its moving-coupled fraction remains close to v12/v13.

v14 best_contact subset readout:

| subset | contact | moving coupled |
|---|---:|---:|
| chairs | 15.45 cm | 0.646 |
| imhd | 35.52 cm | 0.103 |
| neuraldome | 33.87 cm | 0.248 |
| omomo_correct_v2 | 24.64 cm | 0.289 |

Decision: v14 is a partial positive result, not a solution. Use v14
`best_contact.pt` as the current one-shot contact checkpoint, but diagnose its
K-sample capacity before starting another training branch.

v14 K=16 diagnostics:

| selection | oracle mean | saved-best remeasure | moving coupled |
|---|---:|---:|---:|
| distance | 16.80 cm | 17.60 cm | 0.326 |
| composite | 17.17 cm | 17.94 cm | 0.3715 |

v14 K=16 composite by subset:

| subset | contact | moving coupled |
|---|---:|---:|
| chairs | 10.39 cm | 0.764 |
| imhd | 23.28 cm | 0.218 |
| neuraldome | 21.18 cm | 0.314 |
| omomo_correct_v2 | 13.84 cm | 0.424 |

This is a meaningful candidate-pool improvement over v12 composite, especially
on IMHD (`31.95 cm -> 23.28 cm`, max-coupled mean `0.180 -> 0.238`). Composite
selection changed `13/80` picks relative to distance-only; among changed clips,
coupling increased by `0.287` on average at a `2.29 cm` contact cost.

Wandb history is synced locally at
`runs/wandb_logs/wandb_history_genB_v14_sampled_st_contact.csv`. It shows the
decoded auxiliary objective optimizing as intended: train decoded loss drops
from `1.303` to `0.403`, validation decoded loss from `0.898` to `0.425`, and
validation decoded mean-min-dist from `0.564 m` to `0.153 m`. The train-time
best contact checkpoint is epoch 65 (`26.33 cm`, moving coupled `0.308`,
composite `0.3556`); best-val is epoch 70 (`30.45 cm`, coupled `0.280`).
Offline eval of the saved checkpoint gives `27.37 cm` / `0.2765`, so use the
offline summaries for final comparison and wandb for training dynamics.

Visual review of the v14 K=16 best videos changed the next diagnostic target:
composite is visibly better than earlier generations and slightly better than
distance-only, but the result is still not GT-quality. The person/object
misalignment remains obvious enough that the generation should be considered
unacceptable.

New alignment diagnostic:

```bash
python scripts/stage_b_generator/measure_contact_alignment.py \
  --generated-dir runs/eval/stageB_v0_14_bc_k16_composite_oracle/best \
  --gt-dir runs/eval/stageB_v0_14_sampled_st_contact_gt_roundtrip_80/gt_roundtrip \
  --output-dir runs/eval/stageB_v0_14_bc_k16_composite_oracle/alignment_to_gt_roundtrip \
  --detail full
```

Result against GT roundtrip:

| selection | moving contact IoU | moving GT-contact recall | correct GT-part recall | same-part local pos error |
|---|---:|---:|---:|---:|
| distance K=16 | 0.4505 | 0.5468 | 0.2305 | 46.42 cm |
| composite K=16 | 0.4472 | 0.5438 | 0.2378 | 46.32 cm |

The self-check on GT roundtrip vs itself gives moving IoU/recall/correct-part
recall `1.0` and same-part local position error `0.0`, so the diagnostic is
calibrated. Interpretation: the current distance/composite scores are now
metric-gaming-prone. Composite modestly improves correct body-part recall, but
neither selection aligns the generated body to the GT object-local contact
trajectory. Guidance/reranking must become body-part and contact-target aware,
not only "any tracked part near any object point."

### 4. Soft-hard gap diagnostic

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

### 5. RVQ mixed oracle

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

### 6. Subset-specific codebook audit

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

## After v14 Diagnostics

Immediate branch:

- Treat v14 K=16 composite as a strong spatial baseline but not a successful
  interaction baseline: `17.94 cm`, moving-coupled `0.3715`, moving GT-contact
  IoU `0.4472`, correct GT-part recall `0.2378`.
- Use v14 `best_contact.pt` as the base checkpoint for full-RVQ sample-time
  guidance through decoded motion, but guide against the predicted/conditioned
  contact body part and object-local target trajectory rather than the old
  any-part min-distance.
- Keep `measure_contact_alignment.py` as the paired readout with contact
  distance and temporal coupling. A real win must improve GT-aligned contact,
  not only lower mean-min distance.

Secondary diagnostics, only if reranked samples fail visually or hard subsets
remain unacceptable:

- Subset-specific VQ audit, especially IMHD.
- Full-detail v14 visual/hard-case review, especially the clips still above
  `40 cm` or with near-zero moving coupling.

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
