# PIANO Progress

Compact project memory as of 2026-04-30.

## Current Snapshot

Recent Stage B implementation:

```text
v14 sampled straight-through contact trajectory loss
```

Current Stage B best server checkpoint identifier. The local workspace may only
have eval summaries, not the `.pt` file:

```text
runs/training/generator_v14_sampled_st_contact/best_contact.pt
```

Matched 80-clip contact eval:

| Row | mean_min_dist_per_frame |
|---|---:|
| GT original | 13.09 cm |
| GT VQ roundtrip | 18.47 cm |
| v12 w02 best_val full | 31.82 cm |
| v13 target-trajectory best_val full | 31.57 cm |
| v14 sampled-ST best_contact full | 27.37 cm |
| v14 sampled-ST best_val full | 30.77 cm |
| v14 sampled-ST final full | 31.12 cm |
| v12 w02 K=16 distance oracle | 17.93 cm |
| v12 w02 K=16 composite oracle | 18.08 cm |
| v14 best_contact K=16 distance oracle | 16.80 cm |
| v14 best_contact K=16 composite oracle | 17.17 cm |

Bottom line: v14 is the first clear single-sample contact improvement after
the v12/v13 plateau, and its K=16 candidate pool is also better. v14
best_contact improves full contact from the old `31-32 cm` band to `27.37 cm`;
v14 K=16 distance reaches `16.80 cm`, and v14 K=16 composite re-measures at
`17.94 cm` with moving-coupled `0.3715`, beating the v12 K=16 composite
coupling (`0.351`). The remaining bottleneck is choosing or guiding these
samples reliably, not proving contact candidates exist.

2026-04-30 follow-up analysis: v14 sampled-ST helps both one-shot spatial
contact and K=16 candidate quality, but ordinary single-sample generation still
rarely selects the coupled candidates. The next main path should be full-RVQ
sample-time guidance or a reranking baseline over v14 samples, plus visual
review of the v14 K=16 composite outputs.

2026-04-30 visual/alignment update: v14 K=16 composite looks much better than
earlier generations and slightly better than distance-only, but still visibly
fails GT-quality object contact. The new CPU-only diagnostic
`scripts/stage_b_generator/measure_contact_alignment.py` compares generated
samples to GT roundtrip contact in object-local coordinates. Composite K=16 has
moving contact IoU `0.4472`, moving GT-contact recall `0.5438`, correct
GT-body-part recall `0.2378`, and same-GT-part object-local position error
`46.32 cm`. Distance-only is almost identical (`0.4505` IoU, `0.2305` correct
part recall, `46.42 cm` local error). The GT self-check gives IoU, recall, and
correct-part recall `1.0`, plus same-part local position error `0.0`,
validating the diagnostic. Conclusion: the current distance/composite metrics are
insufficient; next guidance must be body-part and contact-target aware.

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

- Active training-loss bottleneck: the generator must learn temporally bound
  manipulation, not only spatial proximity.
- Current implementation includes residual `z_int` conditioning and decoded
  contact auxiliary loss through full RVQ prediction.
- v13 replaced the old arbitrary min-distance decoded loss with a part-specific
  object-local contact-target trajectory objective plus a moving-object
  local-velocity term.
- v14 keeps the target-trajectory objective but uses all-mask MaskGIT/CFG
  first-step logits, straight-through Gumbel hard codebook lookups, and full
  residual RVQ rollout for the decoded auxiliary path.
- Main training script: `src/piano/training/train_generator.py`.
- Main model wrapper: `src/piano/models/motion_generator.py`.
- Decoded contact loss: `src/piano/training/decoded_contact_loss.py`.
- Current training runner:
  `scripts/stage_b_generator/run_v14_sampled_st_contact.sh`.
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
| v13 target trajectory loss | best_val 31.57 cm, coupled 0.265 | soft target improves internally but hard sampled output stays on v12 line |
| v13 RVQ diagnostics | soft_train_full 14.78 cm / 0.443 coupled; mixed_pred_all 33.50 cm | soft-hard and base-token path gaps are real |
| v14 sampled-ST loss | best_contact 27.37 cm, coupled 0.277 | first single-sample contact gain, but temporal binding remains weak |
| v14 wandb history | best_contact selected at epoch 65: train-time 26.33 cm / coupled 0.308 | decoded aux optimized, contact/coupling remain stochastic |
| v14 K=16 oracle | distance 16.80 cm; composite 17.17 cm, remeasured 17.94 cm / coupled 0.3715 | v14 candidate pool improves; selection/guidance is now the main lever |
| v14 RVQ diagnostics | mixed_pred_all 29.31 cm vs v13 33.50; pred base + GT residual 29.81 vs 35.92 | sampled/base path improved, residual bottleneck remains |
| v14 contact alignment | composite moving IoU 0.447; correct GT-part recall 0.238; local part error 46 cm | spatial contact metrics are being gamed; use part/target-aware guidance |

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

1. Implement full-RVQ sample-time guidance through decoded motion on top of v14
   `best_contact.pt`, using predicted/conditioned contact body part and
   object-local target trajectory rather than any-part min-distance.
2. Evaluate with contact distance, temporal coupling, and
   `measure_contact_alignment.py`; a real improvement must raise GT-aligned
   moving contact and correct body-part recall.
3. If guidance cannot select the aligned candidates reliably, build a reranker
   around part/target-aware features, not only the old composite score.

Secondary diagnostics:

- Subset-specific codebook audit if IMHD remains poor after v14 K-sample search.
- Full visual review of v14 best_contact hard cases before treating the contact
  gain as semantically meaningful.

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
