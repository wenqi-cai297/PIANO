# Stage B Compact Analysis

Merged on 2026-04-29 from the previous dated Stage B notes. This is the durable
source for Stage B memory.

## Current Conclusion

Stage B has a real object/interaction control signal. One-shot C2b/v12 and
v13 sampling sat around `31-32 cm` full contact on the matched 80-clip eval.
v14 sampled-ST is the first clear single-sample contact improvement, with
`best_contact` at `27.37 cm`. More importantly, v14 K=16 reaches the GT VQ
roundtrip band: distance oracle `16.80 cm`, composite oracle `17.17 cm`
(`17.94 cm` after saved-best remeasure) versus GT roundtrip `18.47 cm`.

Do not continue blind parameter sweeps. v14 confirms that hard/ST sampled-path
training improves spatial contact and candidate-pool quality, but K64
alignment-aware selection shows the candidate pool usually still lacks truly
GT-aligned interaction samples. v14 K=16 composite raises moving-coupled frame
fraction to `0.3715`, above v12 composite `0.351`; K64 alignment then drops to
`0.3339` while only modestly improving local-position error. v15's
alignment/coupling loss plus full-RVQ final-stage guidance has now been run
and is negative/neutral, so the active next path is v16 deterministic mirror
doubling on top of v15, not another contact-loss weight, checkpoint-selection,
or pure reranking sweep.

2026-04-30 visual/alignment update: v14 K=16 composite looks much better than
earlier generations, but the user's visual assessment is correct that it is
still not GT-quality interaction. A new contact-alignment diagnostic shows
distance/composite metrics are now too weak: v14 K=16 composite has moving
contact IoU `0.4472`, moving GT-contact recall `0.5438`, correct GT body-part
recall `0.2378`, and same-part object-local position error `46.32 cm` against
GT roundtrip. The next guidance/reranking objective must use the conditioned
contact body part and object-local contact target, not only any-part distance.

2026-04-30 K64 alignment update: v14 K=64 alignment-aware selection is not a
successful interaction baseline. It remeasures at `18.71 cm` contact, moving
coupled `0.3339`, moving contact IoU `0.4516`, correct GT-part recall
`0.2496`, and same-part object-local error `40.30 cm`. This is only a modest
local-alignment gain over K=16 composite and it worsens coupling. Across all 64
candidates, the per-clip best primary alignment error is still `37.0 cm` and
the best moving same-part recall is only `0.165` on average. The v14
distribution usually does not contain a GT-aligned manipulation sample to pick,
especially for NeuralDome and OMOMO.

2026-04-30 v15 result update: v15 adds wrong-part margin, contact-segment
consistency, strict alignment checkpointing, and full-RVQ final-stage guidance,
but the generated distribution did not improve. `best_contact` raw full is
`27.62 cm`, moving contact IoU is `0.3804`, moving correct GT-part recall is
`0.1684`, and moving same-part local error is `55.09 cm`. `full_guided` worsens
contact to `31.57 cm`. Local `piano` visualization under
`runs/visualizations/stageB_v0_15_bc_review` shows the same failure: trolley
and suitcase clips still have clear human-object offsets, sometimes worse
after guidance.

2026-04-30 v16 implementation update: the data loader now supports
`AugmentConfig.mirror_duplicate`. When enabled, train-set length doubles:
even indices load the original clip and odd indices load a forced mirrored
copy. The mirror path uses the existing Stage-B-correct augmentation logic for
joints, HumanML3D `motion_263`, contact body-part labels/targets, text L/R
swaps, and world-frame object pose. New config/runner:
`configs/training/generator_v16_alignment_mirror.yaml` and
`scripts/stage_b_generator/run_v16_alignment_mirror.sh`. Evidence: the
[MoMask CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_MoMask_Generative_Masked_Modeling_of_3D_Human_Motions_CVPR_2024_paper.pdf)
§4 states HumanML3D/KIT-ML are augmented by mirroring, and the official
[HumanML3D README](https://github.com/EricGuo5513/HumanML3D) describes
doubling the dataset with mirrored motions and text-side left/right keyword
replacement.

2026-05-01 v16 server result update: partial positive but does not close the
K-oracle gap. Raw `best_contact full` 26.79 cm vs v15 27.62 cm, `full_guided`
28.91 cm vs v15 31.57 cm (the v15 guidance-induced contact regression is
largely fixed). On `final` ckpt, moving correct GT-part recall is `0.1990` —
the highest non-oracle value in the project. Same-part local error 53.49
(bc) / 53.23 (bv) / 52.91 (final) cm vs v15's 55.09 / 59.92 / 54.24.
However v16 is still ~8-13 cm short of the v14 K=16 distance oracle
(17.60 cm) and v14 K=64 alignment oracle (40.30 cm local error). Per restart
prompt rule, mirror-doubling is worth keeping but not the breakthrough; next
iteration moves to a different mechanism, not another data/loss-weight knob.

2026-05-01 v17 implementation update: per-step decoded-geometric guidance
landed locally. Inference-time only — runs on the existing v16 (or v14/v15)
`best_contact.pt` unchanged. Closes MaskControl ICCV 2025's `each_iter` half
of the recipe that PIANO had been running only the post-hoc half of since
v15. Design + ablation plan:
[analyses/2026-05-01_per_step_guidance_design.md]. Entry points:
`src/piano/inference/contact_guidance.py::_generate_with_per_step_guidance`,
new CLI `--per-step-iters` etc. in
`scripts/stage_b_generator/qual_eval.py`, runner
`scripts/stage_b_generator/run_v17_per_step_guidance.sh`. Default v17-C:
`per_step_iters=10`, `guidance_steps=0`.

2026-05-01 v17-C result: largest single-sample contact gain in project
history. On v16 `best_contact.pt`, no retraining: contact `21.77 cm`
(v16 raw 26.79, v16 `full_guided` 28.91, K=16 distance oracle 17.60),
moving_coupled `0.3428` (v14 K=64 alignment oracle 0.3339 — **v17-C
single-sample beats the K=64 oracle on coupling**), moving IoU `0.4388`
(v14 K=16 0.4472), correct-part recall `0.2020` (v14 K=16 0.2378),
same-part object-local position error `46.13 cm` (v14 K=16 composite
oracle 46.32 cm — **v17-C single-sample matches the K=16 oracle on local
error**). Design success threshold 2 of 3 pass; correct-part recall ≥ 0.22
is the only miss (1.8 pp short). Per-step inner loop flips 60.67% of base
tokens vs naive baseline (vs 0–30% for post-hoc-only). Detail and
per-step trace analysis in `analyses/2026-05-01_v17_per_step_result.md`.

2026-05-01 v17-D / v17-E next: `scripts/stage_b_generator/run_v17_sweep.sh`
runs v17-D stacked (per_step=10, post_hoc=30; canonical MaskControl
recipe), v17-E.20 (per_step=20), v17-E.50 (per_step=50) back-to-back.
Targets the remaining 1.8 pp correct-part recall gap.

2026-04-29 literature/code review update: the current bottleneck is best framed
as a sample-time geometric feedback problem. C2b optimizes a soft, differentiable
decoded contact path, but generation uses discrete MaskGIT/Gumbel base tokens,
residual RVQ generation, and VQ decode. The v0.12 sweep shows the soft surrogate
can improve without improving generated contact. This aligns with recent
methods: OMOMO uses hand positions as an explicit intermediate target,
InterDiff injects an interaction correction step during diffusion, CHOIS adds
contact guidance during sampling, and MaskControl combines differentiable
sampling consistency with inference-time logits/codebook optimization.

## Current Best

Best one-shot contact checkpoint identifier on the server. The local workspace may only contain
eval summaries:

```text
runs/training/generator_v14_sampled_st_contact/best_contact.pt
```

80-clip matched eval:

| condition | value |
|---|---:|
| GT original | 13.09 cm |
| GT roundtrip | 18.47 cm |
| v12 w02 best_val full | 31.82 cm |
| v13 best_val full | 31.57 cm |
| v14 best_contact full | 27.37 cm |
| v14 best_contact text_only | 57.82 cm |
| v14 best_contact swap | 74.79 cm |
| v15 best_contact full | 27.62 cm |
| v15 best_contact full_guided | 31.57 cm |
| K=16 best-of-K oracle | 17.93 cm |
| K=16 composite oracle | 18.08 cm |
| v14 K=16 distance oracle | 16.80 cm |
| v14 K=16 composite oracle | 17.17 cm |
| v14 K=64 alignment oracle | 17.92 cm |

Saved v14 K=16 composite samples re-measured with `measure_contact_distance.py`
at `17.94 cm`, with moving-coupled `0.3715`. Interpretation: `z_int` matters
and v14 improves both single-sample spatial contact and the K-sample candidate
pool.

Saved v14 K=64 alignment samples re-measure at `18.71 cm`, moving-coupled
`0.3339`, moving IoU `0.4516`, correct GT-part recall `0.2496`, and same-part
local error `40.30 cm`. Interpretation: alignment-aware selection improves the
object-local patch error slightly, but the candidate pool itself lacks enough
truly aligned manipulation samples. The remaining problem is now distribution
alignment/coupling, not merely selecting among existing v14 samples.

## External Evidence That Shaped The Design

Object-conditioned HOI generators consistently condition on object trajectory,
geometry, affordance, or contact information, not text alone. Reviewed methods
included CG-HOI, HOI-Diff, CHOIS, InterDiff, OMOMO, Move as You Say, Text2HOI,
and InterMask. The durable design lesson was to feed object pose/geometry into
the generator-side condition, which led to v0.2 object pose channels.

Adapter/control literature shaped the gamma and trainable-copy experiments:

- ControlNet, GLIGEN, IP-Adapter, T2I-Adapter, and LLaMA-Adapter established
  zero-init/gated adapter patterns, but their exact recipes differ.
- OmniControl, InterControl, and MotionLCM show that motion control often uses
  frozen/trainable-copy control branches rather than a tiny inline gamma gate.
- MaskControl-style guidance motivated B3, but PIANO's residual RVQ sampling
  made base-logit guidance directionally unstable.

The updated MaskControl lesson is narrower: base-only post-hoc guidance was too
weak/unstable, but the stronger published recipe optimizes logits or embeddings
through a decoded geometric loss at each unmasking stage and at the final stage.
v15 implemented the final-stage full-RVQ version for PIANO's RVQ stack, but
the synced result did not validate it. Per-step MaskGIT guidance remains a
possible follow-up only if a future branch shows stronger decoded constraints
can improve alignment without worsening contact.

Practical rule: use upstream MoMask/VQ/recovery functions. Avoid custom
motion recovery or VQ decode logic unless no upstream function exists.

## Architecture Timeline

| version | change | result |
|---|---|---|
| v0.1 | base MaskTransformer `z_int` adapter | weak visual/object signal |
| v0.2 | added object pose channels to `z_int` | token signal rose, body often in place |
| v0.3 alpha | world/body-frame variants explored | source audit refuted original gamma-stuck framing |
| v0.3 beta/v0.4 | fixed MoMask encoder normalization | major body/token collapse fix |
| v0.5 | longer CE training | CE better, contact worse |
| v0.6 | per-head gamma | canonical 5-clip contact improved to 16.03 |
| v0.7 | mirror augmentation | regressed to 29.50 |
| v0.3-delta | trainable-copy InterControl variant | dead init fixed, final run still regressed |
| B1 | contact-aware checkpointing | exposed CE/contact decoupling |
| B3 | contact-aware base-logit guidance | mixed wins/losses, no stable ship path |
| C1/v0.8 | residual `z_int` conditioning | failed alone |
| C2/v0.9 | decoded contact auxiliary loss | `z_int` became contact-effective |
| C2b/v0.10 | decoded loss through full soft RVQ | improved 20-clip result, not enough |
| v0.11 | gradient/loss diagnostics | weight 0.10 too small |
| v0.12 | decoded-contact weight sweep | all useful ckpts tied near 32 cm |
| K=16 oracle | contact reranking over 16 samples | 17.93 cm best-of-K on 80 clips |
| temporal coupling | kinematic-coupling diagnostic on K=16 best | only 0.323 coupled on moving frames |
| v13 | object-local target-trajectory loss | hard sampling stayed near 31.57 cm / 0.265 coupled |
| v14 | sampled-ST full-RVQ decoded auxiliary path | best_contact improved to 27.37 cm but only 0.277 coupled |
| v14 K=16 | distance 16.80 cm; composite remeasured 17.94 cm / 0.3715 coupled | candidate pool is good enough; selection/guidance is the lever |
| v14 contact alignment | composite moving IoU 0.447; correct GT-part recall 0.238; local part error 46 cm | distance/composite can look good while visual contact remains wrong |
| v14 K=64 alignment | remeasured 18.71 cm / 0.3339 coupled / 0.2496 correct-part recall | higher K and alignment reranking still lack aligned candidates |

## Key Positive Results

### v0.4 normalization fix

MoMask VQ encoder previously received raw motion instead of normalized
HumanML3D features. Fixing this repaired token collapse/body-in-place behavior.
This was the first major Stage B correction.

### B0 contact metric

`measure_contact_distance.py` replaced subjective mp4 viewing as the main
contact metric. On the canonical 5 clips:

- GT original: `10.52 cm`
- GT roundtrip: `11.29 cm`
- v0.4 full: `20.86 cm`

This showed the model gap was larger than the codebook gap on that small set.

### v0.6 per-head gamma

Per-head gamma improved canonical 5-clip contact from v0.4 `20.86 cm` to
`16.03 cm`, mostly through right-hand contact improvements. Useful, but it did
not solve the general task.

### v0.9 decoded contact loss

C2 made `z_int` matter for contact:

- v0.9 best_val full: `29.19 cm`
- text_only: `61.72 cm`
- swap: `64.61 cm`

This was a real control signal. The old best_contact checkpoint for v0.9 was
invalid as strategy evidence because it was selected on only 5 clips.

### v0.10 full-RVQ path

C2b routed decoded contact supervision through soft base logits plus
differentiable residual RVQ rollout before VQ decode. It improved the 20-clip
best_val result to `25.27 cm`, but did not close the gap.

## Negative Results To Preserve

### CE/contact decoupling

v0.5 lowered CE but worsened contact. Later B1 also showed validation CE and
contact checkpoint can diverge. Do not optimize CE alone and expect contact to
improve.

### Mirror augmentation

Mirror augmentation was verified mathematically and by tests, but v0.7
regressed contact. Bilateral data symmetry was not the missing lever.

### Trainable-copy InterControl

The first v0.3-delta attempt had a double-zero-init problem: gamma-zero on top
of connector-zero blocked the gradient path at step 0. After fixing gamma to
start at 1.0 while keeping connector-zero, the run still regressed relative to
v0.6. Do not repeat this architecture swap without a new diagnostic reason.

### B3 base-logit guidance

Contact guidance on base token logits produced both large wins and large losses
depending on clip and residual rerun behavior. Same mechanism caused the best
win and worst failure. Base logits alone are not a stable binding lever.

### C1 residual conditioning alone

v0.8 residual `z_int` conditioning failed:

- full: `43.62 cm`
- text_only: `44.01 cm`
- swap: `49.33 cm`

Residual wiring without a decoded contact objective is not sufficient.

### v0.12 weight sweep

Increasing decoded-contact weight improved gradient share and scalar decoded
loss, but not generated contact:

| weight | decoded grad median | final decoded loss | best full |
|---:|---:|---:|---:|
| 0.20 | 5.57% | 0.1558 | 31.82 |
| 0.30 | 7.97% | 0.1340 | 32.39 |
| 0.50 | 13.02% | 0.1128 | 32.87 |
| 0.80 | 19.27% | 0.0988 | 32.17 |

This is strong evidence against more blind contact-weight tuning.

### K=16 sample oracle

The K-sample oracle sampled 16 variants per matched eval clip from
`runs/training/generator_v12_decoded_contact_w02_diagnostics/best_val.pt` and
selected the lowest contact-distance variant using the existing metric.

| metric | value |
|---|---:|
| single-sample mean | 32.22 cm |
| K=16 sample mean | 31.64 cm |
| K=16 best-of-K mean | 17.93 cm |
| K=16 best-of-K median | 14.50 cm |
| best under 22 cm | 70% |
| best under 25 cm | 80% |

Per-subset:

| subset | single | best-of-K | under 25 cm |
|---|---:|---:|---:|
| chairs | 18.51 | 8.44 | 95% |
| imhd | 42.90 | 29.38 | 70% |
| neuraldome | 37.87 | 21.66 | 60% |
| omomo_correct_v2 | 29.60 | 12.23 | 95% |

This is decisive evidence that the learned distribution already has good
spatial contact modes. The immediate Stage B problem is selecting/guiding
samples for temporally correct manipulation, not making the decoded-contact
loss larger.

### K=16 visual review and temporal coupling

Human review of the saved K=16 best videos:

- Positive: generated bodies are much closer to objects; motion direction often
  broadly aligns with object trajectory; the model clearly uses object
  coordinates.
- Negative: action timing is weakly tied to object motion. For moving objects,
  the body can perform a plausible text action near the object without visibly
  touching it or carrying/pushing it through the object's movement.

`measure_temporal_coupling.py` quantifies this using the same kinematic
coupling criterion as pseudo-label extraction: during moving-object frames, a
body part should stay stable in the object's local frame.

Result on distance-reranked K=16 best:

| metric | value |
|---|---:|
| ordinary mean contact distance | 0.187 m |
| moving-object frame fraction | 0.555 |
| moving frames with any close tracked body part | 0.475 |
| moving frames with kinematic coupling | 0.323 |
| moving frames close but uncoupled | 0.245 |

Subset moving-coupled frame fraction:

| subset | value |
|---|---:|
| chairs | 0.665 |
| imhd | 0.134 |
| neuraldome | 0.277 |
| omomo_correct_v2 | 0.379 |

This confirms the visual assessment. Distance-only contact is not a sufficient
proxy for "the person is manipulating the object." IMHD is the clearest failure
mode.

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
average contact distance worsened by about `0.99 cm`, while moving-coupled
frame fraction improved strongly on those clips. The aggregate gain remains
small because most selected samples are unchanged and the K=16 pool rarely
contains strongly coupled candidates for IMHD.

Offline rescoring of the stored K=16 candidates gives the practical ceiling:

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

Conclusion: more rerank-weight sweeps are unlikely to solve Stage B. The model
needs a temporal-binding mechanism: decoded kinematic-coupling/local-frame
stability loss, contact-target trajectory loss in object-local coordinates, or
full-RVQ sample-time guidance through decoded motion.

v13 implemented the training-loss branch: `mode="target_trajectory"` tracks the
same contact body part against its object-local `contact_target_xyz` and adds a
moving-object local-frame velocity term. v14 moved that objective closer to the
hard sampled path with all-mask first-step logits and straight-through Gumbel
full-RVQ decoding. This improved one-shot contact but did not solve temporal
binding.

## v0.12 Full Table

| ckpt | full | text_only | swap |
|---|---:|---:|---:|
| w02 best_contact | 33.33 | 64.61 | 72.73 |
| w02 best_val | 31.82 | 64.85 | 74.01 |
| w02 final | 32.51 | 61.62 | 70.68 |
| w03 best_contact | 32.39 | 61.88 | 70.60 |
| w03 best_val | 33.83 | 62.16 | 75.89 |
| w03 final | 33.61 | 58.90 | 71.54 |
| w05 best_contact | 32.99 | 64.95 | 76.28 |
| w05 best_val | 32.87 | 59.22 | 73.38 |
| w05 final | 35.49 | 60.84 | 73.00 |
| w08 best_contact | 34.92 | 61.90 | 76.10 |
| w08 best_val | 32.73 | 59.17 | 74.30 |
| w08 final | 32.17 | 59.39 | 72.54 |

Paired bootstrap against the best row did not produce a decisive winner; the
top rows are effectively tied for strategic purposes.

## 80-Clip Decomposition

For v12 w02 best_val:

| subset | GT orig | GT roundtrip | full | codebook gap | model gap |
|---|---:|---:|---:|---:|---:|
| chairs | 12.04 | 12.09 | 19.07 | 0.05 | 6.98 |
| imhd | 8.41 | 22.55 | 42.30 | 14.13 | 19.75 |
| neuraldome | 14.80 | 19.30 | 33.63 | 4.50 | 14.33 |
| omomo_correct_v2 | 17.11 | 19.95 | 32.27 | 2.84 | 12.32 |

This revises the earlier small-set conclusion: codebook/roundtrip is not always
negligible, especially for IMHD. The model gap is still larger overall, but
subset-specific representation issues must be checked.

## Current Implementation Artifacts

Configs:

- `configs/training/generator_v16_alignment_mirror.yaml`
- `configs/training/generator_v15_alignment_guided.yaml`
- `configs/training/generator_v13_target_trajectory_contact.yaml`
- `configs/training/generator_v14_sampled_st_contact.yaml`
- `configs/training/generator_v10_full_rvq_decoded_contact_aux.yaml`
- `configs/training/generator_v11_diagnostics.yaml`
- `configs/training/generator_v12_decoded_contact_w03_diagnostics.yaml`
- generated sweep configs from
  `scripts/stage_b_generator/run_v12_contact_weight_sweep.sh`

Script:

- `scripts/stage_b_generator/run_v16_alignment_mirror.sh`
- `scripts/stage_b_generator/run_v15_alignment_guided.sh`
- `scripts/stage_b_generator/run_v14_sampled_st_contact.sh`
- `scripts/stage_b_generator/run_v13_rvq_diagnostics.sh`
- `scripts/stage_b_generator/diagnose_rvq_paths.py`
- `scripts/stage_b_generator/run_v13_target_trajectory.sh`
- `scripts/stage_b_generator/run_v12_contact_weight_sweep.sh`
- `scripts/stage_b_generator/k_sample_oracle.py` (`--selection-metric composite`
  selects by contact distance plus moving-object coupling; `alignment` selects
  by paired GT-contact alignment when a GT directory is provided)
- `scripts/stage_b_generator/measure_temporal_coupling.py`
- `scripts/stage_b_generator/measure_contact_alignment.py`

Logging/eval hygiene:

- v16 inherits v15's decision metrics and enables deterministic train-set mirror
  doubling; validation and offline eval are still unaugmented.
- v15 training reports decision metrics by default: total/base/
  residual/decoded losses, aggregate base/residual accuracy, decoded target
  position/velocity, wrong-part margin, segment consistency, soft-distance,
  gate magnitudes, and contact eval alignment/contact/coupling metrics.
- v13/v14 training runners now also export the new alignment keys when present.
- v13 training originally reported only decision metrics by default: total/base/
  residual/decoded losses, aggregate base/residual accuracy, decoded target
  position/velocity/soft-distance, gate magnitudes, and contact eval
  `composite_contact_score`, `mean_min_dist`, `moving_close`,
  `moving_coupled`, `close_uncoupled`, `n_clips`.
- Per-RVQ-layer residual metrics and gradient-norm probes are disabled in
  `generator_v13_target_trajectory_contact.yaml`; re-enable them only for a
  targeted residual-stack or gradient-scale diagnostic.
- `qual_eval.py`, `measure_contact_distance.py`,
  `measure_temporal_coupling.py`, and `diagnose_rvq_paths.py` write compact
  JSON by default. Use `--detail full` or `--summary-detail full` when per-clip
  rows are needed.

Core code:

- `src/piano/training/train_generator.py`
- `src/piano/training/decoded_contact_loss.py`
- `src/piano/training/contact_eval.py`
- `src/piano/models/motion_generator.py`
- `src/piano/models/backbones/momask_adapter.py`
- `src/piano/data/eval_sampling.py`

## Next Work

### 1. Temporal-binding mechanism

Question: can training or inference make the generated distribution contain
samples where the relevant body part is stable in the moving object's local
frame?

Current implementation:

- v13 decoded `target_trajectory` loss: part-specific object-local contact
  target tracking plus moving-object local-frame velocity supervision.
- Use contact distance and temporal coupling as paired readouts.

v13 outcome after syncing results:

| run | contact | moving close | moving coupled | close but uncoupled |
|---|---:|---:|---:|---:|
| v12 w02 best_val | 31.82 cm | 0.296 | 0.264 | 0.140 |
| v13 best_val | 31.57 cm | 0.334 | 0.265 | 0.171 |
| v14 best_contact | 27.37 cm | 0.343 | 0.277 | 0.172 |
| v15 best_contact | 27.62 cm | 0.338 | 0.284 | 0.174 |
| v12 K=16 composite | 18.08 cm | 0.473 | 0.351 | 0.222 |
| GT roundtrip | 18.47 cm | - | - | - |

The v13 soft decoded auxiliary objective did optimize internally
(`decoded_contact_aux_mean_min_dist` about `0.413 -> 0.125`), but hard
sampled output stayed on the same 31-32 cm / 0.26 moving-coupled line as v12.
The v13 RVQ diagnostic then showed `soft_train_full` at `14.78 cm` and
moving-coupled `0.443`, while sampled/mixed prediction paths stayed much worse
(`mixed_pred_all` `33.50 cm`; `mixed_pred_base_gt_residual` `35.92 cm`).

v14/C2c addressed that gap by taking decoded-aux base logits from the all-mask
MaskGIT/CFG first step and using straight-through Gumbel hard codebook lookups
through the full residual RVQ path. Result: v14 best_contact is a partial
positive result for spatial contact, but not temporal binding.

Wandb history for v14 is synced at
`runs/wandb_logs/wandb_history_genB_v14_sampled_st_contact.csv`. It confirms the
sampled-ST auxiliary objective optimized: train decoded loss `1.303 -> 0.403`,
validation decoded loss `0.898 -> 0.425`, and validation decoded mean-min-dist
`0.564 m -> 0.153 m`. The train-time contact selector chose epoch 65
(`26.33 cm`, moving coupled `0.308`, composite `0.3556`), while best-val is
epoch 70 (`30.45 cm`, coupled `0.280`). Offline evaluation of the synced
`best_contact` output measured `27.37 cm` and coupled `0.2765`, so generation
sampling variance is nontrivial but the strategic conclusion is unchanged.

v14 best_contact subset readout, using the matched 80 clips:

| subset | contact | moving coupled |
|---|---:|---:|
| chairs | 15.45 cm | 0.646 |
| imhd | 35.52 cm | 0.103 |
| neuraldome | 33.87 cm | 0.248 |
| omomo_correct_v2 | 24.64 cm | 0.289 |

Worst v14 one-shot temporal-binding failures remain IMHD baseball/suitcase and
NeuralDome tennis/baseball/trashcan/keyboard cases. The large IMHD suitcase
outlier improved from v13 `120.33 cm` to v14 `67.24 cm`, but remains a hard
one-shot failure.

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

The candidate pool improved compared with v12 composite. IMHD is still hard
but much better spatially (`31.95 cm -> 23.28 cm`) and has slightly better
coupled-candidate capacity (max-coupled mean `0.180 -> 0.238`; clips with any
candidate >=0.5: `2/20 -> 3/20`). Composite selection changed `13/80` clips,
increasing coupled fraction by `0.287` on changed clips at a `2.29 cm` contact
cost.

v14 RVQ-path diagnostics:

| condition | v13 contact | v14 contact | v13 coupled | v14 coupled |
|---|---:|---:|---:|---:|
| mixed_pred_all | 33.50 cm | 29.31 cm | 0.289 | 0.296 |
| mixed_pred_base_gt_residual | 35.92 cm | 29.81 cm | 0.234 | 0.279 |
| mixed_gt_base_pred_residual | 27.30 cm | 26.96 cm | 0.349 | 0.306 |
| hard_train_argmax_full | 22.47 cm | 28.98 cm | 0.349 | 0.324 |
| soft_train_full | 14.78 cm | 29.41 cm | 0.443 | 0.324 |

Interpretation: v14 improved the standard generated/mixed predicted paths,
especially the predicted-base path. It did not improve the old teacher-forced
soft diagnostic path, which is expected because v14 trains the decoded aux from
the all-mask generation-entry path instead. The remaining bottleneck is not
"no candidates"; it is sample-time selection/guidance and residual quality.

v14 visual review and GT-alignment diagnostic:

```bash
python scripts/stage_b_generator/measure_contact_alignment.py \
  --generated-dir runs/eval/stageB_v0_14_bc_k16_composite_oracle/best \
  --gt-dir runs/eval/stageB_v0_14_sampled_st_contact_gt_roundtrip_80/gt_roundtrip \
  --output-dir runs/eval/stageB_v0_14_bc_k16_composite_oracle/alignment_to_gt_roundtrip \
  --detail full
```

| selection | moving contact IoU | moving GT-contact recall | correct GT-part recall | same-part local pos error |
|---|---:|---:|---:|---:|
| distance K=16 | 0.4505 | 0.5468 | 0.2305 | 46.42 cm |
| composite K=16 | 0.4472 | 0.5438 | 0.2378 | 46.32 cm |

The GT roundtrip self-check gives moving IoU/recall/correct-part recall `1.0`
and same-part local position error `0.0`. Composite is slightly better on
correct body-part recall, but it does not solve GT contact alignment. This
confirms the qualitative review: even a low mean-min distance can be produced
by the wrong part, wrong phase, or wrong object-local patch.

v14 K=64 alignment-aware oracle:

| selection | contact remeasure | moving coupled | moving IoU | correct GT-part recall | same-part local pos error |
|---|---:|---:|---:|---:|---:|
| K=16 distance | 17.60 cm | 0.3260 | 0.4505 | 0.2305 | 46.42 cm |
| K=16 composite | 17.94 cm | 0.3715 | 0.4472 | 0.2378 | 46.32 cm |
| K=64 alignment | 18.71 cm | 0.3339 | 0.4516 | 0.2496 | 40.30 cm |

K=64 alignment selection improves same-part local position error by roughly
`6 cm` relative to K=16 composite, but it does not improve moving contact IoU
and it reduces moving-coupled frame fraction. Candidate-capacity analysis over
all 64 samples per clip shows why:

| subset | best primary alignment over K64 | best moving same-part recall over K64 | best distance over K64 | clips primary <25 cm | clips recall >=0.5 |
|---|---:|---:|---:|---:|---:|
| all | 37.01 cm | 0.165 | 13.89 cm | 35% | 9% |
| chairs | 21.24 cm | 0.430 | 7.58 cm | 75% | 56% |
| imhd | 35.27 cm | 0.147 | 19.83 cm | 25% | 5% |
| neuraldome | 55.53 cm | 0.068 | 17.13 cm | 20% | 0% |
| omomo_correct_v2 | 36.00 cm | 0.147 | 11.02 cm | 20% | 0% |

Several hard clips have a close distance candidate but still no aligned
contact candidate, for example `subject03_trolleycase_1083_3` has best distance
`4.19 cm` but best primary alignment `117.21 cm`. This is strong evidence that
the old contact distance can be gamed by the wrong part/patch and that pure
K-sample reranking is not enough.

Immediate diagnostic implementation:

- `scripts/stage_b_generator/diagnose_rvq_paths.py`
- `scripts/stage_b_generator/run_v13_rvq_diagnostics.sh`
- `scripts/stage_b_generator/measure_contact_alignment.py`

These produce `soft_train_full`, `hard_train_argmax_full`,
`hard_train_argmax_gt_residual`, `mixed_gt_all`, `mixed_pred_all`,
`mixed_gt_base_pred_residual`, and `mixed_pred_base_gt_residual`, then score
them with the existing contact-distance and temporal-coupling scripts.

Immediate next work:

- Run v16 mirror-doubled alignment training and compare raw `full` vs guided
  `full_guided`. First beat v15 raw (`27.62 cm`, moving IoU `0.3804`, correct
  GT-part recall `0.1684`, local error `55.09 cm`), then compare against the
  stronger v14 K=16 composite baseline (`17.94 cm`, coupled `0.3715`, moving
  IoU `0.4472`, correct GT-part recall `0.2378`) and v14 K=64 alignment
  baseline (`18.71 cm`, coupled `0.3339`, moving IoU `0.4516`, correct GT-part
  recall `0.2496`, local error `40.30 cm`).
- The objective uses predicted/conditioned `contact_body_part`,
  object-local `contact_target_xyz` trajectory, and local-frame coupling
  together rather than the any-part min-distance objective that the current
  metrics can game.

### 2. Hard-case follow-up

Question: which remaining failures are search failures, representation
failures, or metric/semantic mismatches?

Start with the clips that remain above `25 cm` after K=16, especially:

- `20230901_wangwzh_suitcase_suitcase_lefthand_carry_3_0`: `116.76 cm`
- `subject03_tennis_926`: `59.05 cm`
- `20230825_wangwzh_bat_bat_lefthand_swing_11_0_0`: `58.81 cm`
- `20230901_wangwzh_suitcase_suitcase_lift_0_1328`: `46.54 cm`
- `subject04_baseball_0`: `37.20 cm`

Try higher K or targeted diagnostics on IMHD/NeuralDome before changing
training.

### 3. Soft-hard gap diagnostic

Question: is the decoded auxiliary loss optimistic relative to hard generation?

Rows:

- soft full-RVQ decoded contact from the training auxiliary path;
- argmax base + residual rollout;
- sampled base + residual rollout;
- ordinary eval contact.

Decision:

- If soft is much better than hard, switch from pure soft expectation to
  ST-Gumbel/DES-style consistency or sample-time logits/embedding optimization.
- If soft is also bad, change the learned contact representation/distribution.

### 4. RVQ mixed oracle

Question: where does the error enter the discrete RVQ stack?

Rows:

- GT all-RVQ;
- predicted all-RVQ;
- GT base + predicted residual;
- predicted base + GT residual.

### 5. Subset codebook audit

Question: which clips/subsets have large GT original -> GT roundtrip drift?

Start with IMHD because its 80-clip codebook gap is `14.13 cm`.

## Decision Rules

- K-sample oracle worked spatially: the model can place bodies near objects.
- Distance-only visual review failed temporally: do not ship distance-only
  reranking as Stage B.
- Composite reranking only modestly improves coupling: do not spend the next
  main iteration on rerank-weight sweeps.
- K64 alignment oracle shows v14 K-sample capacity remains semantically weak:
  do not spend the next main iteration on larger K or rerank-weight tuning
  alone.
- Next main path: change the learned distribution with alignment/coupling
  supervision or use full-RVQ decoded-motion guidance that optimizes part,
  target, and coupling jointly.
- If base mixed oracle fails: base MaskTransformer/conditioning is the target.
- If residual mixed oracle fails: residual/full-RVQ prediction is the target.
- If GT roundtrip fails by subset: address representation/codebook for that
  subset before more generator-only training.

## Source Trail

Previous detailed notes merged into this document:

- `2026-04-26_stageB_design.md`
- `2026-04-27_object_conditioning_review.md`
- `2026-04-27_adapter_source_review.md`
- `2026-04-27_motion_conditioning_2024_2026.md`
- `2026-04-27_v0_3_root_cause_research.md`
- `2026-04-27_v0_5_premise_review.md`
- `2026-04-27_v0_7_outcome.md`
- `2026-04-28_b1_b3_iteration_log.md`
- `2026-04-28_c1_c2_landing_review.md`
- `2026-04-28_v0_9_outcome.md`
- `2026-04-29_v12_weight_sweep_global_review.md`

Primary external families referenced across those notes:

- object-conditioned HOI: CG-HOI, HOI-Diff, CHOIS, InterDiff, OMOMO,
  Move as You Say, Text2HOI, InterMask;
- control/adapters: ControlNet, GLIGEN, IP-Adapter, T2I-Adapter,
  LLaMA-Adapter;
- motion control: OmniControl, InterControl, MotionLCM, MaskControl.
