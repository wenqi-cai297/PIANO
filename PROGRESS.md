# PIANO Progress

Compact project memory as of 2026-04-29.

## Current Snapshot

Recent Stage B diagnostic code commit:

```text
c4b2f50 Add Stage B temporal coupling rerank diagnostic
```

Tracked git state was clean before the K-sample result documentation update.

Current Stage B best server checkpoint identifier. The local workspace may only
have eval summaries, not the `.pt` file:

```text
runs/training/generator_v12_decoded_contact_w02_diagnostics/best_val.pt
```

Matched 80-clip contact eval:

| Row | mean_min_dist_per_frame |
|---|---:|
| GT original | 13.09 cm |
| GT VQ roundtrip | 18.47 cm |
| v12 w02 best_val full | 31.82 cm |
| v12 w02 best_val text_only | 64.85 cm |
| v12 w02 best_val swap | 74.01 cm |
| v12 w02 K=16 distance oracle | 17.93 cm |
| v12 w02 K=16 composite oracle | 18.08 cm |

Bottom line: Stage B has a real `z_int` signal, and the v12 generator
distribution already contains close-to-object samples. The remaining gap is
temporal/physical binding: distance reranking puts the body near the object,
but composite K=16 only modestly improves moving-object coupling. This is now
evidence that the K=16 sample pool itself rarely contains strongly coupled
manipulation, especially on IMHD.

2026-04-29 follow-up analysis: the problem is likely not missing contact
supervision. The current loss already includes base CE, residual CE, and a
decoded contact auxiliary term through the full soft RVQ path. The failure mode
is more specific: the soft/teacher-forced decoded contact surrogate improves,
but the discrete sample-time path (MaskGIT base sampling, residual generation,
VQ decode) remains near the same contact band. External HOI/control work points
to closing this sample-time geometric loop with reranking, guidance, hard/ST
sampling consistency, or explicit intermediate contact/joint representations.

## Stage Status

Stage 1 pseudo-labels:

- Current InterAct label track: v11.
- Important fields: contact body part, closest-surface target xyz,
  3-class phase, support state, object pose in canonical/body frame.
- Durable doc: `analyses/pseudo_label_pipeline.md`.

Stage A Interaction Predictor:

- Shipped state: v6.
- Predictor of record: server checkpoint `runs/training/predictor/final.pt`
  (local sync may only include eval JSONs).
- Durable doc: `analyses/stageA_design.md`.
- Revisit only if downstream diagnostics show Stage A labels/predictions are
  the limiting factor.

Stage B Motion Generator:

- Active sample-selection/guidance bottleneck.
- Current implementation includes residual `z_int` conditioning and decoded
  contact auxiliary loss through full soft RVQ prediction.
- Main training script: `src/piano/training/train_generator.py`.
- Main model wrapper: `src/piano/models/motion_generator.py`.
- Decoded contact loss: `src/piano/training/decoded_contact_loss.py`.
- Current sweep runner:
  `scripts/stage_b_generator/run_v12_contact_weight_sweep.sh`.
- Current no-retrain diagnostic runner:
  `scripts/stage_b_generator/k_sample_oracle.py`.
- Durable doc: `analyses/stageB_compact.md`.

Stage C Joint Finetune:

- Not started.
- Do not start until the Stage B reranking/guidance baseline passes metric and
  visual review.

## Stage B Evidence Timeline

| Step | Result | Decision |
|---|---|---|
| v0.1 | initial `z_int` adapter, weak visual effect | needed object pose |
| v0.2 | object pose added, token signal up but body mostly in place | inspect MoMask path |
| v0.3/v0.4 | MoMask encoder normalization bug fixed | first major correction |
| B0 | contact metric introduced; v0.4 full 20.86 vs GT roundtrip 11.29 on canonical 5 | measure contact, not mp4 vibes |
| v0.5 | lower CE but worse contact | CE is misaligned |
| v0.6 | per-head gamma improved canonical 5 to 16.03 | useful but limited |
| v0.7 | mirror augmentation regressed to 29.50 | data symmetry not the fix |
| v0.3-delta | trainable-copy variant regressed after dead-init fix | architecture swap exhausted |
| B1 | contact checkpointing exposed CE/contact decoupling | useful diagnostic, not enough |
| B3 | inference guidance mixed wins/losses | base logits unstable |
| C1/v0.8 | residual `z_int` alone: 43.62 full | not enough |
| C2/v0.9 | decoded contact aux: 29.19 full on 20 clips | real control signal |
| C2b/v0.10 | full-RVQ path: 25.27 full on 20 clips | partial gain |
| v0.11 | diagnostics showed weight 0.10 had small gradient share | motivated sweep |
| v0.12 | weights 0.20/0.30/0.50/0.80 all near 32 cm on 80 clips | stop blind weight sweeps |
| K=16 oracle | best-of-K 17.93 cm on 80 clips | reranking/guidance becomes main path |
| K=16 visual review | body is near object but weakly synchronized to object motion | distance-only reranking is insufficient |
| temporal coupling metric | moving coupled frame frac 0.323 | optimize/rerank for coupling, not only distance |
| K=16 composite oracle | coupled frac 0.351, contact 18.08 cm | only modest gain; K=16 pool lacks enough coupled samples |

## v0.12 Details

Runner:

```bash
bash scripts/stage_b_generator/run_v12_contact_weight_sweep.sh
```

Default weights:

```text
decoded_contact_aux.weight = 0.20, 0.30, 0.50, 0.80
```

Best 80-clip result by full contact:

| Rank | checkpoint | full |
|---:|---|---:|
| 1 | w02 best_val | 31.82 |
| 2 | w08 final | 32.17 |
| 3 | w03 best_contact | 32.39 |
| 4 | w02 final | 32.51 |

Gradient diagnostic:

| weight | decoded grad median | final decoded loss | best full |
|---:|---:|---:|---:|
| 0.20 | 5.57% | 0.1558 | 31.82 |
| 0.30 | 7.97% | 0.1340 | 32.39 |
| 0.50 | 13.02% | 0.1128 | 32.87 |
| 0.80 | 19.27% | 0.0988 | 32.17 |

Conclusion: decoded-contact surrogate optimization is working mechanically, but
it does not translate monotonically to sample-time contact.

K-sample oracle on v12 w02 best_val:

| metric | value |
|---|---:|
| single-sample mean | 32.22 cm |
| K=16 sample mean | 31.64 cm |
| K=16 best-of-K mean | 17.93 cm |
| K=16 best-of-K median | 14.50 cm |
| best under 22 cm | 70% |
| best under 25 cm | 80% |

Saved best samples re-measured with `measure_contact_distance.py` at
`18.70 cm`, close to the oracle score. This confirms the selected saved output
is in the GT VQ roundtrip band.

Per-subset K=16 best-of-K:

| subset | single | best-of-K |
|---|---:|---:|
| chairs | 18.51 | 8.44 |
| imhd | 42.90 | 29.38 |
| neuraldome | 37.87 | 21.66 |
| omomo_correct_v2 | 29.60 | 12.23 |

IMHD remains the hardest subset; the worst outlier is
`20230901_wangwzh_suitcase_suitcase_lefthand_carry_3_0` at `116.76 cm` even
after K=16 reranking.

Visual review of the K=16 best samples: the positive signal is real
object-position conditioning. The body is usually near the object and often
oriented/moving in the same broad direction. The failure is stronger: body
motion is often only colocated with the object trajectory, not temporally bound
to it. In object-moving clips, the person may perform a plausible action near
the object while the object's move timing does not match and no stable contact
is visible.

New diagnostic:

```bash
python scripts/stage_b_generator/measure_temporal_coupling.py \
  --input-dir runs/eval/stageB_v0_12_w02_bv_k16_oracle/best \
  --output-dir runs/eval/stageB_v0_12_w02_bv_k16_oracle/temporal_coupling
```

Result on K=16 distance-reranked best:

| metric | value |
|---|---:|
| ordinary mean contact distance | 0.187 m |
| moving-object frame fraction | 0.555 |
| moving frames with any close tracked body part | 0.475 |
| moving frames with kinematic coupling | 0.323 |
| moving frames close but uncoupled | 0.245 |

Subset coupling:

| subset | moving coupled frame frac |
|---|---:|
| chairs | 0.665 |
| imhd | 0.134 |
| neuraldome | 0.277 |
| omomo_correct_v2 | 0.379 |

This supports the user's visual assessment: distance-only contact is not a
strong enough proxy for "the person is actually manipulating the object."

Composite K=16 reranking result:

| metric | distance K=16 | composite K=16 |
|---|---:|---:|
| contact mean | 17.93 cm | 18.08 cm |
| contact median | 14.50 cm | 14.74 cm |
| under 22 cm | 70% | 70% |
| under 25 cm | 80% | 80% |
| moving coupled frame frac | 0.323 | 0.351 |
| close but uncoupled moving frac | 0.245 | 0.222 |

Composite reranking changed only `12/80` selected samples. Among changed clips,
average contact distance worsened by only `0.99 cm`, while moving-coupled frame
fraction improved by `0.554`. The aggregate gain is small because most clips
kept the same sample and IMHD has weak coupled candidates in the K=16 pool.

Offline rerank over the stored K=16 candidate scores shows the ceiling:

| selection rule | contact mean | moving coupled frac |
|---|---:|---:|
| distance-only | 17.93 cm | 0.325 |
| current composite | 18.08 cm | 0.354 |
| high coupling weight ~1.0 | 19.36 cm | 0.386 |
| max-coupled oracle | 20.67 cm | 0.390 |

Per-subset max-coupled capacity within K=16:

| subset | max-coupled mean | contact at max-coupled | clips with any >=0.5 |
|---|---:|---:|---:|
| chairs | 0.838 | 6.99 cm | 3/3 moving clips |
| imhd | 0.180 | 37.53 cm | 2/20 |
| neuraldome | 0.368 | 27.91 cm | 5/17 |
| omomo_correct_v2 | 0.456 | 18.27 cm | 7/20 |

Conclusion: tuning rerank weights is not enough. The model distribution needs
stronger temporal-binding generation or training; IMHD/baseball/suitcase cases
are the clearest blockers.

Subset decomposition for v12 w02 best_val:

| subset | GT orig | GT roundtrip | full | codebook gap | model gap |
|---|---:|---:|---:|---:|---:|
| chairs | 12.04 | 12.09 | 19.07 | 0.05 | 6.98 |
| imhd | 8.41 | 22.55 | 42.30 | 14.13 | 19.75 |
| neuraldome | 14.80 | 19.30 | 33.63 | 4.50 | 14.33 |
| omomo_correct_v2 | 17.11 | 19.95 | 32.27 | 2.84 | 12.32 |

The old "codebook is negligible" conclusion only held on the canonical 5 clips.
On 80 clips, IMHD has a large roundtrip/codebook issue.

## Next Work

Immediate:

1. Do not continue rerank-weight sweeps as the main path.
2. Add a decoded kinematic-coupling / local-frame stability objective or an
   inference-time full-RVQ coupling guidance step.
3. Use composite reranking as a diagnostic/readout, not as the final Stage B
   solution.

Secondary diagnostics:

- Soft-hard gap diagnostic if composite reranking still exposes visual cheating
  or instability.
- RVQ mixed oracle if outliers suggest base/residual token bottlenecks.
- Subset-specific codebook audit if IMHD remains poor after higher-K search.

## Environment

Server:

```bash
cd /media/gpu-server-1/4TB_for_data/Cai/PIANO/PIANO
conda activate piano
```

Local workspace:

```text
e:\Project\2026-04-13
```

Local tests should use the `piano` conda environment when possible.
