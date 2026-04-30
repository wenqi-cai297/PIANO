# PIANO Plan

Compact action plan as of 2026-05-01.

## Immediate Priority

v17-C per-step decoded-geometric guidance is the new Stage B state-of-the-art
single-sample result, achieved without retraining (runs on the v16
`best_contact.pt` ckpt unchanged). On the matched 80-clip eval:

- contact `21.77 cm` (single-sample), `−5.02 cm` vs v16 raw, `−7.14 cm` vs
  v16 `full_guided`, `+3.83 cm` to v14 K=16 distance oracle (best-of-16).
- moving same-part object-local position error `46.13 cm`, **matching the
  v14 K=16 composite oracle (46.32 cm)** in single-sample.
- moving_coupled `0.3428`, **beating the v14 K=64 alignment oracle (0.3339)**
  in single-sample.

Result detail: `analyses/2026-05-01_v17_per_step_result.md`. The remaining
single design-threshold gap is correct-part recall `0.2020` vs target
`≥ 0.22` (1.8 pp short). Decision-rule outcome: proceed to v17-D (stacked
per-step + post-hoc) and v17-E (per-step iter sweep), targeting the
remaining correct-part gap.

Background: Stage B single-sample generation is no longer completely stuck at
the old `31-32 cm` contact band: v14 sampled-ST `best_contact` reaches
`27.37 cm` full contact on the matched 80-clip eval. This is a real
spatial-contact gain, but GT VQ roundtrip is still `18.47 cm`, and temporal
coupling barely moves (`0.2765` moving-coupled frame fraction vs v13 `0.2653`
and v12 `0.2639`).

Current diagnosis: `z_int` is active, and v14 proves that making the decoded
auxiliary path harder and closer to sampling improves both one-shot spatial
contact and the K-sample candidate pool. However, ordinary one-shot sampling
still often fails to bind to the object's motion, and K=64 alignment-aware
selection shows the v14 pool usually lacks truly GT-aligned manipulation
samples. v15 then tested the direct alignment-loss/full-RVQ-guidance idea and
did not improve the learned raw distribution: `best_contact` full is
`27.62 cm`, essentially tied with v14, while moving correct GT-part recall is
only `0.1684` and same-part local error is `55.09 cm`.

Latest evaluated branch: v14 keeps v13's object-local `contact_target_xyz`
trajectory loss, but takes decoded-aux logits from the all-mask MaskGIT/CFG
first step and decodes with straight-through Gumbel hard codebook samples
through the full residual RVQ path.

Latest implemented branch, pending server results: v16 keeps the v15 alignment
objective but changes the train data to deterministic mirror doubling
(`augmentation.mirror_duplicate=true`). This matches the MoMask/HumanML3D
mirrored-data assumption and is deliberately separate from the older stochastic
v0.7 `mirror_prob=0.5` experiment.

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

v14 K=64 alignment-aware oracle:

| selection | saved-best remeasure | moving coupled | moving IoU | correct GT-part recall | same-part local pos error |
|---|---:|---:|---:|---:|---:|
| K=16 distance | 17.60 cm | 0.3260 | 0.4505 | 0.2305 | 46.42 cm |
| K=16 composite | 17.94 cm | 0.3715 | 0.4472 | 0.2378 | 46.32 cm |
| K=64 alignment | 18.71 cm | 0.3339 | 0.4516 | 0.2496 | 40.30 cm |

The K=64 alignment oracle is a partial negative result. It lowers same-part
object-local position error by about `6 cm` relative to K=16 composite, but
does not improve moving contact IoU and worsens moving-coupled frame fraction.
The candidate-capacity check is more important: the per-clip minimum primary
alignment error over all 64 candidates is still `37.0 cm` on average, and the
best moving same-part recall available in the K=64 pool is only `0.165` on
average. Only about `9%` of clips with finite moving contact recall have any
candidate reaching recall >= `0.5`; NeuralDome and OMOMO have none. This means
the v14 distribution usually does not contain a GT-aligned manipulation sample,
even with K=64.

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

### 4a. v17 per-step decoded-geometric guidance (v17-C shipped, v17-D/E next)

Implemented 2026-05-01 as an inference-time addition. No retraining.
Replaces the baseline MaskGIT loop with a re-rolled version that runs N AdamW
inner steps on the predicted logits at each MaskGIT iteration before commit,
using a relaxed-decode geometric loss with frozen baseline residuals.

**v17-C result (per-step only, `per_step_iters=10`, `guidance_steps=0`):**
contact `21.77 cm` / coupled `0.3428` / IoU `0.4388` / correct-part `0.2020`
/ same-part local `46.13 cm`. Beats every v15/v16 baseline by clear margins.
Same-part local error matches the v14 K=16 composite oracle in single-sample;
coupling beats the v14 K=64 alignment oracle in single-sample. Per-step inner
loop flips 60.67% of base tokens vs naive baseline. See
`analyses/2026-05-01_v17_per_step_result.md`.

**Next runs (v17-F Gumbel sweep)**: launched via
`scripts/stage_b_generator/run_v17f_gumbel_sweep.sh`. Adds Gumbel-Softmax
relaxation to the per-step inner loop (the last unmatched MaskControl
recipe diff, source-verified 2026-05-01 from `exitudio/ControlMM`):

| variant | per_step | gumbel | role |
|---|---:|---:|---|
| v17-F.10 | 10 | 1.0 | canonical MaskControl `each_iter`; ship candidate |
| v17-F.20 | 20 | 1.0 | does Gumbel + bigger budget compound? |
| v17-C-ng | 10 | 0.0 | sanity: must reproduce v17-C 21.77 cm |
| v17-E.20-ng | 20 | 0.0 | sanity: must reproduce v17-E.20 18.62 cm |

Detail: `analyses/2026-05-01_v17_diagnostics_and_gumbel.md`.

**Earlier runs (v17-D + v17-E sweep, 2026-05-01)**: launched via
`scripts/stage_b_generator/run_v17_sweep.sh`. Three eval conditions:

| variant | per_step_iters | guidance_steps | EVAL_PREFIX | hypothesis |
|---|---:|---:|---|---|
| v17-D stacked | 10 | 30 | `stageB_v0_17_v16bc_stacked` | post-hoc on top of per-step adds another 1–3 cm contact + 0.5–2 pp correct-part; canonical MaskControl recipe |
| v17-E.20 | 20 | 0 | `stageB_v0_17_v16bc_per_step_iters20` | does doubling the per-step inner budget close the 1.8 pp correct-part gap? |
| v17-E.50 | 50 | 0 | `stageB_v0_17_v16bc_per_step_iters50` | saturation check; skip if v17-E.20 ≈ v17-C |

Decision rules after sync:

- v17-D > v17-C on correct-part → v17-D becomes ship config.
- v17-E.20 ≈ v17-C → 10 iters is saturated; drop v17-E.50 from comparison.
- v17-E.50 > v17-E.20 → consider MaskControl's full 100 (cost: ~7 min added
  per condition; manageable).
- correct-part stays below 0.22 across all v17-D/E variants → pivot to
  OMOMO-style hand-position intermediate target as the next training-time
  branch.

**Optional**: `PER_STEP_START_STEP=2` ablation if a subset of clips with
initially-low loss (Sub1475, suitcase_lefthand_push) regresses under
per-step. Defer until v17-D/E reveals whether this is correlated.

Entry points:

- `src/piano/inference/contact_guidance.py::_generate_with_per_step_guidance`
  re-rolled MaskGIT loop with the per-step inner optimisation hook.
- `src/piano/inference/contact_guidance.py::_decode_with_relaxed_masked_base`
  differentiable decode with hard committed + soft masked + frozen residual.
- `src/piano/inference/contact_guidance.py::_precompute_residual_emb_sum`
  one-time residual codebook lookup used as the inner loop's frozen context.
- `scripts/stage_b_generator/qual_eval.py` new CLI:
  `--per-step-iters / --per-step-lr / --per-step-temperature /
  --per-step-start-step`. Stacks with the existing post-hoc
  `--guidance-steps`.
- `scripts/stage_b_generator/run_v17_per_step_guidance.sh` runner. Default
  v17-C: per-step only, `PER_STEP_ITERS=10`, `GUIDANCE_STEPS=0`. Runs against
  `runs/training/generator_v16_alignment_mirror/best_contact.pt` by default;
  override `SOURCE_RUN_DIR=` to swap ckpts.
- `tests/test_contact_guidance_per_step.py` 3 CPU tests; signature smoke
  in `tests/test_contact_guidance.py::test_guide_with_contact_signature_accepts_new_kwargs`.

Ablation matrix:

| run | per-step | post-hoc | what it tests |
|---|---:|---:|---|
| v17-A baseline | 0 | 0 | sanity / re-baseline raw generation |
| v17-B post-hoc only | 0 | 30 | reproduce v16 `full_guided` |
| v17-C per-step only | 10 | 0 | does per-step alone move contact? |
| v17-D stacked | 10 | 30 | full MaskControl recipe; ceiling check |
| v17-E budget sweep | {20, 50, 100} | 0 | does more per-step iters help further? |

Decision rule:

- v17-C beats v17-B on contact distance + correct-part recall + same-part
  local error → go to v17-D + v17-E.
- v17-C ≈ v17-B → pivot to OMOMO-style hand-position intermediate target
  as the next training-time branch.
- v17-C worse than v17-B → diagnose with `guidance_trace.json::per_clip[*]
  .info.per_step` before declaring per-step dead.

Success threshold (rough): match v14 K=16 distance oracle (17.60 cm) within
≤ 5 cm raw, with correct-part recall ≥ 0.22 and same-part local error
≤ 48 cm.

Risks: residual approximation drift (frozen baseline residuals stale wrt
post-guidance base), early-step soft-decode meaningless (most positions
masked at low step indices), AdamW init-scale calibration. See §5 of the
design doc.

### 4. v15 result and v16 mirror-doubled branch

v15 has been run and is not a win. Local synced metrics:

| row | contact | moving coupled | moving IoU | correct moving GT-part recall | same-part local error |
|---|---:|---:|---:|---:|---:|
| v15 bc full | 27.62 cm | 0.2837 | 0.3804 | 0.1684 | 55.09 cm |
| v15 bc full_guided | 31.57 cm | 0.2991 | 0.3998 | 0.1603 | 59.95 cm |
| v15 bv full | 30.73 cm | 0.2811 | 0.3617 | 0.1606 | 59.92 cm |
| v15 bv full_guided | 29.50 cm | 0.2854 | 0.3749 | 0.1538 | 54.98 cm |
| v15 final full | 29.68 cm | 0.2733 | 0.3653 | 0.1697 | 54.24 cm |
| v15 final full_guided | 31.01 cm | 0.2705 | 0.3865 | 0.1681 | 56.40 cm |

Visual review using local `piano` env wrote videos to
`runs/visualizations/stageB_v0_15_bc_review/{full,full_guided}`. Trolley and
suitcase hard cases still have obvious human-object offsets; `full_guided`
does not rescue them and can increase separation.

v16 is now the next server run. It keeps the v15 objective and changes only the
training data path to deterministic mirror doubling.

Runner:

```bash
bash scripts/stage_b_generator/run_v16_alignment_mirror.sh
```

Primary code/config changes:

- `src/piano/data/dataset.py`: `AugmentConfig.mirror_duplicate` doubles train
  dataset length and pairs each source clip with a forced mirrored copy.
- `src/piano/training/train_generator.py`: forwards `mirror_duplicate` from
  YAML to `HOIDataset`.
- `configs/training/generator_v16_alignment_mirror.yaml`: enables
  `augmentation.enabled=true`, `mirror_prob=0.0`, and
  `mirror_duplicate=true`.
- `src/piano/training/decoded_contact_loss.py`: adds `part_margin_weight`,
  `part_margin_m`, `segment_consistency_weight`, and
  `segment_consistency_moving_only` to the target-trajectory decoded loss.
- `src/piano/training/contact_eval.py`: logs strict GT-part/object-local
  alignment metrics and computes `alignment_contact_score`.
- `src/piano/inference/contact_guidance.py`: adds
  `guidance_layers="full_rvq"` so sampling-time guidance can optimize the full
  generated RVQ stack, not only base logits.
- `configs/training/generator_v15_alignment_guided.yaml`: baseline v15 config
  retained for reproducibility; v16 copies its loss/monitoring settings.

Decision criteria for success:

- raw v16 `best_contact` must beat v15 raw (`27.62 cm`) and improve strict
  alignment toward the v14 K64 alignment reference (`0.2496` moving correct
  GT-part recall, `40.30 cm` same-part local error).
- guided `full_guided` should improve alignment without destroying contact
  distance/coupling; if it repeats v15's contact/local-error regression, do
  not keep tuning full-RVQ final-stage guidance.
- If v16 still matches v15/v14, data symmetry is not the main blocker and the
  next route should be a stronger sampling/training mechanism, not another
  mirror or loss-weight sweep.

### 5. Soft-hard gap diagnostic

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

- Treat v14 K=16 composite and K=64 alignment as strong diagnostic baselines,
  not successful interaction baselines. K=64 alignment reaches `18.71 cm`
  contact, moving-coupled `0.3339`, moving GT-contact IoU `0.4516`, correct
  GT-part recall `0.2496`, and `40.30 cm` local error.
- Do not spend the next iteration on more K/rerank-weight tuning alone. K=64
  shows the v14 candidate distribution itself lacks enough aligned samples,
  especially outside chairs.
- Run v16 mirror-doubled training next. It keeps the v15 alignment objective
  but fixes the remaining MoMask/HumanML3D data mismatch by pairing each
  training clip with a deterministic mirrored copy.
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
