# PIANO Current Recommendation

Compact recommendation memo as of 2026-04-30.

## Executive Position

Do not continue tuning decoded-contact loss weight or checkpoint selection as
the main Stage B strategy. v0.12 shows the training surrogate can be
strengthened without moving one-shot generated contact below the same roughly
`32 cm` band, and v14 shows that even a harder sampled-ST decoded path mainly
improves spatial contact rather than temporal binding.

The K-sample oracle changed the active plan: the current generator distribution
contains close-to-object samples. K=16 best-of-K reaches `17.93 cm`, close to
GT VQ roundtrip `18.47 cm`.

The visual review then sharpened the diagnosis: distance-only reranking is not
enough. The body is near the object and roughly aware of its coordinates, but
the action is often not temporally bound to the object's motion.

The composite K=16 reranker confirms this is not just a rerank-weight issue:
moving-coupled frame fraction rises only from `0.323` to `0.351`, while contact
stays near GT roundtrip (`17.93 cm` -> `18.08 cm`). Even offline max-coupled
selection over the same K=16 candidates reaches only about `0.390`.

Current framing: this is not simply "contact loss too weak" and not primarily
"the model cannot generate contact." The failure is that ordinary sampling and
distance-only selection do not reliably choose samples where the body part moves
with the object during manipulation.

v14 is a partial positive result: `best_contact` reaches `27.37 cm` one-shot
contact, improving over v12/v13, but moving-coupled frame fraction is only
`0.277`. The next step is not another loss-weight sweep; it is to test whether
v14 improved the K-sample candidate pool and whether the RVQ soft/hard gaps
remain.

## Evidence

v12 matched 80-clip eval:

| checkpoint | full |
|---|---:|
| w02 best_val | 31.82 |
| w08 final | 32.17 |
| w03 best_contact | 32.39 |
| w02 final | 32.51 |

Top checkpoints are too close to make checkpoint selection a path to GT
roundtrip (`18.47 cm`).

Decoded-contact weight sweep:

| weight | decoded grad median | final decoded loss | best full |
|---:|---:|---:|---:|
| 0.20 | 5.57% | 0.1558 | 31.82 |
| 0.30 | 7.97% | 0.1340 | 32.39 |
| 0.50 | 13.02% | 0.1128 | 32.87 |
| 0.80 | 19.27% | 0.0988 | 32.17 |

The surrogate behaves as intended, but sample-time contact does not follow.

K=16 oracle on v12 w02 best_val:

| metric | value |
|---|---:|
| single-sample mean | 32.22 cm |
| K=16 sample mean | 31.64 cm |
| K=16 best-of-K mean | 17.93 cm |
| K=16 best-of-K median | 14.50 cm |
| best under 22 cm | 70% |
| best under 25 cm | 80% |

Per-subset K=16 best-of-K:

| subset | best-of-K |
|---|---:|
| chairs | 8.44 cm |
| imhd | 29.38 cm |
| neuraldome | 21.66 cm |
| omomo_correct_v2 | 12.23 cm |

This is strong evidence for a reranking/guidance branch. IMHD still needs
targeted analysis.

Temporal-coupling diagnostic on the distance-reranked K=16 best samples:

| metric | value |
|---|---:|
| ordinary mean contact distance | 0.187 m |
| moving-object frame fraction | 0.555 |
| moving frames with any close tracked body part | 0.475 |
| moving frames with kinematic coupling | 0.323 |
| moving frames close but uncoupled | 0.245 |

By subset, moving-coupled frame fraction is `0.665` chairs, `0.134` IMHD,
`0.277` NeuralDome, and `0.379` OMOMO. This backs the visual finding: contact
distance alone can select "near the object" samples without selecting true
manipulation.

Composite K=16:

| metric | distance K=16 | composite K=16 |
|---|---:|---:|
| contact mean | 17.93 cm | 18.08 cm |
| moving coupled frame frac | 0.323 | 0.351 |
| close but uncoupled moving frac | 0.245 | 0.222 |

K=16 pool capacity check:

| subset | max-coupled mean | clips with any >=0.5 |
|---|---:|---:|
| chairs | 0.838 | 3/3 moving |
| imhd | 0.180 | 2/20 |
| neuraldome | 0.368 | 5/17 |
| omomo_correct_v2 | 0.456 | 7/20 |

This is evidence against spending the next iteration on more rerank-weight
tuning. The distribution must produce more temporally coupled samples.

v14 sampled-ST result:

| checkpoint | full | moving coupled |
|---|---:|---:|
| best_contact | 27.37 cm | 0.277 |
| best_val | 30.77 cm | 0.276 |
| final | 31.12 cm | 0.274 |

v14 best_contact by subset:

| subset | contact | moving coupled |
|---|---:|---:|
| chairs | 15.45 cm | 0.646 |
| imhd | 35.52 cm | 0.103 |
| neuraldome | 33.87 cm | 0.248 |
| omomo_correct_v2 | 24.64 cm | 0.289 |

Wandb history confirms the v14 auxiliary objective optimized rather than
stalling: train decoded loss went `1.303 -> 0.403`, validation decoded loss
`0.898 -> 0.425`, and validation decoded mean-min-dist `0.564 m -> 0.153 m`.
Train-time contact selection picked epoch 65 (`26.33 cm`, moving coupled
`0.308`), while offline eval of the synced `best_contact` output is
`27.37 cm` / `0.277`. Treat this as sampling variance around the same partial
positive result.

v14 K=16 diagnostics now show the candidate pool is genuinely stronger:

| selection | oracle mean | saved-best remeasure | moving coupled |
|---|---:|---:|---:|
| distance | 16.80 cm | 17.60 cm | 0.326 |
| composite | 17.17 cm | 17.94 cm | 0.3715 |

Composite v14 K=16 beats the previous v12 composite coupling (`0.351`) while
staying in the GT roundtrip contact band. IMHD improves substantially
(`31.95 cm -> 23.28 cm` under composite K=16), though it remains the hardest
subset.

RVQ diagnostics show v14 improved the generated/mixed predicted paths:
`mixed_pred_all` moves `33.50 -> 29.31 cm`, and predicted base + GT residual
moves `35.92 -> 29.81 cm`. The old teacher-forced `soft_train_full` diagnostic
gets worse (`14.78 -> 29.41 cm`), but that is not the v14 supervised aux path;
v14 trained the all-mask generation-entry ST path.

Visual review and contact alignment now show why v14 K=16 still is not
acceptable as generated HOI. Against GT roundtrip, `measure_contact_alignment.py`
reports:

| selection | moving contact IoU | moving GT-contact recall | correct GT-part recall | same-part local pos error |
|---|---:|---:|---:|---:|
| distance K=16 | 0.4505 | 0.5468 | 0.2305 | 46.42 cm |
| composite K=16 | 0.4472 | 0.5438 | 0.2378 | 46.32 cm |

The GT self-check gives perfect temporal/body-part scores and `0.0` same-part
local position error. Therefore the user's visual assessment is correct:
distance/composite can reach the GT roundtrip distance band while still missing
the GT contact part, timing, and object-local patch by a large margin.

## What Already Worked

- MoMask encoder normalization fix: repaired token/body collapse.
- Contact-distance metric: exposed the real failure mode.
- Per-head gamma: gave the main early architecture improvement.
- Decoded contact auxiliary loss: made `z_int` matter for contact.
- Full-RVQ decoded-contact path: partial improvement on 20 clips.

## What Is Not Worth Repeating Blindly

- More CE training.
- Mirror augmentation.
- Trainable-copy InterControl variant without new diagnosis.
- Base-logit-only guidance as the main solution.
- Residual `z_int` conditioning alone.
- Larger decoded-contact weights.
- Picking a different v12 checkpoint and expecting a large jump.

## Recommended Next Implementation

Build the next Stage B path under `scripts/stage_b_generator/` and reuse
library helpers from `src/piano/`:

- use existing eval sampling so the 80-clip set is matched;
- use upstream MoMask/VQ decode and `recover_from_ric` paths;
- use existing contact-distance code rather than reimplementing geometry;
- use the pseudo-label extractor's kinematic-coupling criterion for
  moving-object binding;
- output small `summary.json` files, not large generated arrays unless needed.

Current scripts:

- `scripts/stage_b_generator/run_v14_sampled_st_contact.sh`
- `scripts/stage_b_generator/run_v13_target_trajectory.sh`
- `scripts/stage_b_generator/k_sample_oracle.py`
- `scripts/stage_b_generator/measure_temporal_coupling.py`
- `scripts/stage_b_generator/measure_contact_alignment.py`

`run_v14_sampled_st_contact.sh` is the latest train/eval runner. It wraps the
v13 runner with v14 defaults. `k_sample_oracle.py` remains the key next
diagnostic readout and supports `--selection-metric composite`.

Decision rules:

- K-sample oracle succeeded spatially: the model can place bodies near objects.
- Distance-only visual review failed temporally: spatial proximity is not enough.
- Composite reranking only modestly improved coupling: do not spend another
  main iteration on rerank-weight sweeps.
- v14 K=16 candidate quality is spatially strong but semantically misaligned:
  prioritize full-RVQ logits/embedding optimization or decoded-motion
  sample-time guidance using the predicted contact body part and object-local
  target trajectory.
- Use v14 K=16 composite as the baseline to beat: `17.94 cm`, moving coupled
  `0.3715`, moving contact IoU `0.4472`, correct GT-part recall `0.2378`.
- If v14 RVQ diagnostics still show a large base-token gap: focus
  MaskTransformer/base conditioning.
- RVQ mixed oracle identifies base bottleneck: focus MaskTransformer/base
  conditioning.
- RVQ mixed oracle identifies residual bottleneck: focus residual/full-RVQ
  conditioning and loss.
- Codebook audit shows subset roundtrip failure: treat representation as a
  subset-specific bottleneck, especially IMHD.
