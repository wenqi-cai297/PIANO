# Stage B Compact Analysis

Merged on 2026-04-29 from the previous dated Stage B notes. This is the durable
source for Stage B memory.

## Current Conclusion

Stage B has a real object/interaction control signal, but the current C2b
training strategy is stuck around `32 cm` full contact on the matched 80-clip
eval. GT VQ roundtrip is `18.47 cm`, so there is still a large generation gap.

Do not continue blind parameter sweeps. Run no-retrain diagnostics:

1. K-sample oracle.
2. Soft-hard gap diagnostic.
3. RVQ mixed oracle.
4. Subset-specific codebook audit.

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

Best checkpoint identifier on the server. The local workspace may only contain
eval summaries:

```text
runs/training/generator_v12_decoded_contact_w02_diagnostics/best_val.pt
```

80-clip matched eval:

| condition | value |
|---|---:|
| GT original | 13.09 cm |
| GT roundtrip | 18.47 cm |
| full | 31.82 cm |
| text_only | 64.85 cm |
| swap | 74.01 cm |

Interpretation: `z_int` matters, but prediction/decoding still loses about
`13.35 cm` beyond GT roundtrip.

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
That supports a future full-RVQ logits/embedding optimization branch if the
K-sample oracle or soft-hard diagnostic says good contact is reachable by
sample-time search.

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

- `configs/training/generator_v10_full_rvq_decoded_contact_aux.yaml`
- `configs/training/generator_v11_diagnostics.yaml`
- `configs/training/generator_v12_decoded_contact_w03_diagnostics.yaml`
- generated sweep configs from
  `scripts/stage_b_generator/run_v12_contact_weight_sweep.sh`

Script:

- `scripts/stage_b_generator/run_v12_contact_weight_sweep.sh`
- `scripts/stage_b_generator/k_sample_oracle.py`

Core code:

- `src/piano/training/train_generator.py`
- `src/piano/training/decoded_contact_loss.py`
- `src/piano/training/contact_eval.py`
- `src/piano/models/motion_generator.py`
- `src/piano/models/backbones/momask_adapter.py`
- `src/piano/data/eval_sampling.py`

## Next Diagnostics

### 1. K-sample oracle

Question: does the current generator already place good-contact samples in the
distribution?

Expected output:

- best-of-K contact per clip;
- mean/median and per-subset summary;
- comparison to single-sample v12 w02 best_val and GT roundtrip.

### 2. Soft-hard gap diagnostic

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

### 3. RVQ mixed oracle

Question: where does the error enter the discrete RVQ stack?

Rows:

- GT all-RVQ;
- predicted all-RVQ;
- GT base + predicted residual;
- predicted base + GT residual.

### 4. Subset codebook audit

Question: which clips/subsets have large GT original -> GT roundtrip drift?

Start with IMHD because its 80-clip codebook gap is `14.13 cm`.

## Decision Rules

- If K-sample oracle works: add reranking/guidance/scoring.
- If K-sample oracle fails: change the learned distribution.
- If soft-hard gap is large: use hard/ST-Gumbel consistency or full-RVQ
  logits/embedding optimization.
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
