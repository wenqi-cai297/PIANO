# PIANO Progress

Compact project memory as of 2026-05-01.

## Current Snapshot

Recent Stage B implementation:

```text
v16 alignment-aware contact loss + deterministic mirror-doubled training set
```

Current Stage B best evaluated server checkpoint identifier. The local workspace may only
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
| v15 alignment-guided best_contact full | 27.62 cm |
| v15 alignment-guided best_contact full_guided | 31.57 cm |
| v12 w02 K=16 distance oracle | 17.93 cm |
| v12 w02 K=16 composite oracle | 18.08 cm |
| v14 best_contact K=16 distance oracle | 16.80 cm |
| v14 best_contact K=16 composite oracle | 17.17 cm |
| v14 best_contact K=64 alignment oracle | 17.92 cm |

Bottom line: v14 is the first clear single-sample contact improvement after
the v12/v13 plateau, and its K=16 candidate pool is also better. v14
best_contact improves full contact from the old `31-32 cm` band to `27.37 cm`;
v14 K=16 distance reaches `16.80 cm`, and v14 K=16 composite re-measures at
`17.94 cm` with moving-coupled `0.3715`, beating the v12 K=16 composite
coupling (`0.351`). The later K64 alignment oracle shows this is still not an
aligned-HOI distribution: spatially close candidates exist, but GT-aligned
part/patch/timing candidates are too rare.

2026-04-30 follow-up analysis: v14 sampled-ST helps both one-shot spatial
contact and K=16 candidate quality, but ordinary single-sample generation still
rarely selects coupled candidates and K64 alignment selection shows the pool
itself lacks enough aligned ones. v15 tested the direct alignment/guidance
branch and did not solve this, so v16 now tests deterministic mirrored-data
doubling before abandoning this loss family.

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

2026-04-30 K64 alignment-oracle update: selecting among 64 v14 samples with the
alignment-aware score gives oracle contact `17.92 cm` and post-hoc remeasure
`18.71 cm`, but moving-coupled frame fraction drops to `0.3339`. GT-alignment
barely changes in time (`0.4516` moving IoU) and only slightly improves body
part correctness (`0.2496` correct GT-part recall), while same-part local error
improves from `46.32 cm` to `40.30 cm`. Per-candidate capacity is the key
negative result: the best primary alignment error over all K=64 candidates is
still `37.0 cm` on average, and the best moving same-part recall available is
only `0.165`. NeuralDome and OMOMO have zero clips with any K=64 candidate
reaching moving same-part recall >= `0.5`. Conclusion: v14 reranking is close
to exhausted; the distribution itself needs stronger alignment/coupling
training or guidance.

2026-04-30 v15 result update: v15 alignment-guided training is a negative or
at best neutral result. `best_contact` raw full is `27.62 cm`, essentially tied
with v14 `27.37 cm`; the strict GT-alignment readouts are worse than the v14
K-oracle baselines (`0.3804` moving IoU, `0.1684` moving correct GT-part recall,
`55.09 cm` moving same-part local error). Full-RVQ target guidance improves
some temporal-overlap readouts slightly but usually worsens contact/local error:
for best_contact, `full_guided` is `31.57 cm` contact and `59.95 cm` local
error. Local visualization in the `piano` conda env confirms the numbers:
`runs/visualizations/stageB_v0_15_bc_review/{full,full_guided}` still shows
visible human-object offset, and guidance can move the body farther from the
object on trolley/suitcase cases.

2026-04-30 implementation update: v16 has been added for the next server run.
It keeps the v15 alignment objective but changes the data side to deterministic
MoMask/HumanML3D-style mirror doubling: `HOIDataset.__len__` doubles when
`augmentation.mirror_duplicate=true`, even indices are original clips, and odd
indices are forced mirrored copies. Validation/eval remain unaugmented. New
artifacts: `configs/training/generator_v16_alignment_mirror.yaml`,
`scripts/stage_b_generator/run_v16_alignment_mirror.sh`, and
`tests/test_dataset_mirror_duplicate.py`.

2026-05-01 v16 server result: partial positive, doesn't close the K-oracle gap.
Raw `best_contact full` 26.79 cm vs v15 27.62 cm (+0.83 cm); `full_guided`
28.91 cm vs v15 31.57 cm (+2.66 cm, the v15 guidance-induced contact regression
is largely fixed). Moving correct GT-part recall on `final` ckpt is `0.1990`
(highest non-oracle in the project). Same-part local error 53.49 / 53.23 /
52.91 cm across bc/bv/final vs v15's 55.09 / 59.92 / 54.24. Still ~8-13 cm
short of the v14 K=16 distance oracle (17.60 cm) and v14 K=64 alignment oracle
(40.30 cm local error). Decision-gate verdict: mirror-doubling is worth keeping
but is not a breakthrough; per restart prompt rule we now move to a different
mechanism, not another data/loss-weight knob. See
`analyses/2026-05-01_per_step_guidance_design.md`.

⚠️ v16 wandb diagnostic: train-time `contact_alignment_contact_score`,
`contact_composite_contact_score`, and `contact_alignment_moving_same_part_recall`
are degenerate-early-maximum selectors (peak at epochs 5/5/35 respectively
with very small values). The ckpt-of-record uses `contact_mean_min_dist`
which is fine; do not promote any of the three to ship-metric without
adding a sanity floor (e.g. require `contact_mean_min_dist < X` first).

2026-05-01 v17-C result: largest single-sample contact gain in project
history, on the v16 `best_contact.pt` ckpt without retraining. Per-step
decoded-geometric guidance only (no post-hoc):
`full_guided` contact `21.77 cm` (v16 raw 26.79; v16 `full_guided` 28.91;
v14 K=16 composite oracle `17.94 cm`). Moving coupled `0.3428` (v16 raw
0.2734; v14 K=64 alignment oracle 0.3339 — **v17-C single-sample beats the
K=64 oracle on coupling**). Moving IoU `0.4388` (v14 K=16 0.4472).
Moving correct GT-part recall `0.2020` (v14 K=16 0.2378). Same-part
object-local position error `46.13 cm` (v14 K=16 oracle 46.32 cm —
**v17-C single-sample matches the K=16 composite oracle on local error**).
Design success threshold: 2 of 3 pass (contact ≤ 22.60 cm: pass at 21.77;
local error ≤ 48 cm: pass at 46.13; correct-part ≥ 0.22: miss at 0.2020,
−1.8 pp). Per-step inner loop flips 60.67% of base tokens vs naive
baseline on average, much deeper than the 0–30% flip rate of post-hoc-only
guidance. Detail: `analyses/2026-05-01_v17_per_step_result.md`.

Decision-rule outcome (per design doc §4): "v17-C clearly beats v17-B" →
proceed to v17-D (stacked per-step + post-hoc) and v17-E (per-step budget
sweep at iters ∈ {20, 50}).

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
- v15 adds alignment-aware negatives/segment binding to the same decoded path,
  and evaluates full-RVQ target guidance as a sampling-time correction.
- v16 keeps v15's objective and turns on deterministic mirror duplication for
  training only, matching MoMask/HumanML3D's mirrored-data assumption more
  closely than the old stochastic v0.7 `mirror_prob=0.5` test.
- Main training script: `src/piano/training/train_generator.py`.
- Main model wrapper: `src/piano/models/motion_generator.py`.
- Decoded contact loss: `src/piano/training/decoded_contact_loss.py`.
- Current training runner:
  `scripts/stage_b_generator/run_v16_alignment_mirror.sh`.
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
| v14 K64 alignment oracle | remeasured 18.71 cm; coupled 0.334; moving IoU 0.452; correct GT-part recall 0.250; local part error 40 cm | alignment selection gives only modest local-position gain; K64 pool lacks enough aligned samples |
| v15 alignment-guided | best_contact full 27.62 cm; guided 31.57 cm; moving IoU 0.380 and correct GT-part recall 0.168 | negative/neutral; alignment losses did not create GT-quality samples |
| v16 mirror-doubled | deterministic original+mirror train-set duplication implemented and tested locally | next server run; tests MoMask/HumanML3D data assumption without conflating with old stochastic mirror p=0.5 |
| v16 server result | bc full 26.79 cm; bv full 28.13; final full 28.05; final correct GT-part recall 0.1990; same-part local 52.91-53.49 cm | partial positive vs v15; still 8-13 cm short of v14 K-oracle baselines; decision-gate triggers next-mechanism branch (v17 per-step) |
| v17 per-step guidance | inference-time MaskControl-style each_iter logit optimisation, runs on existing v14/v15/v16 ckpts unchanged; v17-C runner + tests landed | next server run; ablation plan in analyses/2026-05-01_per_step_guidance_design.md (v17-A baseline, v17-B post-hoc only, v17-C per-step only, v17-D stacked, v17-E iter sweep) |
| v17-C result | full_guided 21.77 cm contact / 0.3428 coupled / 0.4388 IoU / 0.2020 correct-part / 46.13 cm same-part local | matches v14 K=16 composite oracle on local error; beats K=64 alignment oracle on coupling; advance to v17-D + v17-E |

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

1. Run v17-D + v17-E sweep on the server with the new wrapper:
   `bash scripts/stage_b_generator/run_v17_sweep.sh`. Three back-to-back
   eval conditions on the v16 best_contact ckpt:
   - v17-D stacked: `PER_STEP_ITERS=10 GUIDANCE_STEPS=30` — canonical
     MaskControl recipe (per-step + post-hoc).
   - v17-E.20: `PER_STEP_ITERS=20 GUIDANCE_STEPS=0` — does doubling the
     per-step inner budget close the remaining 1.8 pp correct-part gap?
   - v17-E.50: `PER_STEP_ITERS=50 GUIDANCE_STEPS=0` — saturation check.
2. Sync the three eval matrices back; compare contact / coupled / IoU /
   correct-part / same-part local against v17-C (21.77 / 0.3428 / 0.4388
   / 0.2020 / 46.13). Each condition lands under its own
   `EVAL_PREFIX=stageB_v0_17_v16bc_*` directory tree.
3. Decision branches:
   - v17-D beats v17-C on correct-part recall → ship as the canonical
     v17 config; record alongside v14/v15/v16 in the comparison table.
   - v17-E.20 ≈ v17-E.10 → 10 inner iters is saturated, drop the 50 run.
     If v17-E.50 > v17-E.20 by ≥ 1 cm contact and ≥ 1 pp correct-part →
     consider MaskControl's full 100 with cost analysis.
   - If correct-part recall stays below 0.22 across all v17-D/E variants
     → pivot to OMOMO-style hand-position intermediate target as the
     next training-time branch.
4. Optional follow-up: `PER_STEP_START_STEP=2` ablation to test whether
   skipping early MaskGIT iterations rescues the small subset of clips
   (Sub1475, suitcase_lefthand_push) where per-step inner loss currently
   *increases*. Run only if v17-D/E reveals correlated regression on
   initially-low-loss clips.

Secondary diagnostics:

- Subset-specific codebook audit if IMHD remains poor after v14 K-sample search.
- Full visual review of v16 best_contact hard cases before treating any contact
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
