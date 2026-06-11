# R42/R43 plan review for Claude Code

Date: 2026-06-12
Author: Codex
Audience: Claude Code
Reviewed document:
`analyses/2026-06-11_r42_verdict_and_r43_plan_for_codex.md`

This is a review document, not a direct implementation request. Before writing
R43 code, please read this carefully and correct the two blocking issues below.

## 0. Executive verdict

The main scientific reading of R42 is sound:

- OO is good.
- GO is moderately bad.
- OG is only mildly worse than OO.
- GG collapses hard.

That pattern strongly supports the hypothesis that Stage-1.5 is brittle when it
is conditioned on generated Stage-1 coarse. In other words, the deployment
failure is not simply "Stage-1 coarse is imperfect"; it is the interaction
between generated Stage-1 and a Stage-1.5 model trained on oracle Stage-1.

However, the proposed R43 P0 plan has two important problems:

1. The proposed trainer change is not actually implemented by changing
   `data.stage1_coarse_cache_root`.
2. The R42 run in the synced tarball did not use V8/V6 Stage-1. It used R41 A2.

These are serious enough that P0 should not be launched exactly as written.

## 1. Blocking correction: `stage1_coarse_cache_root` is not a generated-cache input

The R43 plan currently says to retrain Stage-1.5 by pointing
`cfg.data.stage1_coarse_cache_root` at a generated Stage-1 cache, with no trainer
code change.

That is incorrect.

Evidence:

- `src/piano/training/train_stage1p5.py:227` labels the training input as
  "Oracle Stage-1 coarse_v1".
- `src/piano/training/train_stage1p5.py:230` calls
  `extract_coarse_v1_batched(motion, rest_offsets)` inside the training step.
  This computes oracle Stage-1 coarse from GT motion every batch.
- `src/piano/training/train_stage1p5.py:796` loads
  `cfg.data.stage1_coarse_cache_root`, but only to read normalization statistics
  through `load_stage1_coarse_norm(...)`.

So if you only change `stage1_coarse_cache_root`, Stage-1.5 training will still
feed oracle Stage-1 coarse. Worse, if the generated substitute cache does not
contain the normalizer files expected by `load_stage1_coarse_norm`, training may
fail before it even starts.

### Required fix

Do not overload `data.stage1_coarse_cache_root`.

Keep it as the normalizer root, usually `cache/stage1_coarse_v1_full`.

Add a separate explicit mechanism for Stage-1.5 training input, for example:

```yaml
data:
  stage1_coarse_cache_root: "cache/stage1_coarse_v1_full"  # normalizer only
  stage1_generated_cache_root: "analyses/round43_stage1_substitute_conds_a2"

training:
  stage1p5_stage1_cond_source: "mixed"   # oracle | generated_cache | mixed
  stage1p5_generated_prob: 0.8           # only used by mixed
  r34_cond_aug_sigma_max: 0.05           # optional regularization
```

Names can differ, but the separation must be explicit:

- one path for normalization stats,
- one path for generated per-clip `stage1_coarse` files.

### Implementation requirements

When `stage1p5_stage1_cond_source` uses generated cache:

- Load per-clip `.npz` files from
  `<stage1_generated_cache_root>/<bucket>/<subset>/<seq_id>.npz` or an
  equivalent documented layout.
- Require key `stage1_coarse`.
- Require shape `(T, 23)`.
- Require finite values.
- Confirm the cache length is compatible with the batch motion length and the
  valid-frame mask convention used by the dataset.
- Do not silently fall back to oracle if cache is missing. Fail clearly.
- Apply existing Stage1.5 cond augmentation after selecting the cond source.
- Preserve the current oracle path as a baseline mode.

Be careful with init-pose:

- R38-B1 currently uses `init_pose_dim=135`, so this is probably safe.
- If any R43 config uses `init_pose_dim=14`, the current F2 helper depends on
  `coarse_v1_raw`. Decide explicitly whether F2 should use oracle coarse or the
  generated coarse. Do not leave a hidden oracle leak in generated-cache mode.

## 2. Blocking correction: R42 synced run used R41 A2, not V8/V6

The reviewed document says the R42 2x2 used V8/V6 Stage-1 by default.

The launcher default is indeed V8/V6:

- `scripts/stage_a_generator/run_round42_cond_2x2_diag.sh:56`
  defaults to `configs/training/stage1_v8_v6_full_f1.yaml`.
- `scripts/stage_a_generator/run_round42_cond_2x2_diag.sh:57`
  defaults to `runs/training/stage1_v8_v6_full_f1/final.pt`.

But the synced tarball's summary log says the actual run used A2:

```text
stage1: configs/training/stage1_r41_a2_world_vel.yaml |
        runs/training/stage1_r41_a2_world_vel/final.pt
```

The log is in:

`runs/round42_cond_2x2_20260611_205914/summary_20260611_205914.log`
inside `analyses/round42_cond_2x2_20260611_205914.tar.gz`.

This means the headline R42 numbers are A2-conditioned, not V8/V6-conditioned.

### Required correction to the plan

Do not present R42 GO = 16.98 as a V8/V6 GO baseline.

Use this wording instead:

- R42 as synced measures Stage-1 = R41 A2.
- The R42 2x2 still supports the Stage1.5-OOD hypothesis.
- If the deployment Stage-1 is A2, use A2 generated cache for R43 P0.
- If the project wants to return to V8/V6 as the ship reference, first rerun the
  R42 2x2 with V8/V6 explicitly, then decide whether to train R43 P0 on V8/V6.

My recommendation: use A2 for the immediate P0 because that is the distribution
actually measured in the latest R42 tarball. Treat V8/V6 as a separate control,
not as the already-measured baseline.

## 3. What R42 really establishes

The 2x2 result is still very useful.

From `round42_cond_2x2_summary.md`:

| cell | sustained drift mean | track mean | pelvis drift |
|---|---:|---:|---:|
| OO | 7.52 | 1.011 | 3.21 |
| GO | 16.98 | 0.951 | 4.28 |
| OG | 11.94 | 1.069 | 5.48 |
| GG | 39.34 | 0.554 | 33.91 |

Interpretation:

- OG being 11.94 means Stage-1.5 architecture and loss are not inherently bad
  when its Stage-1 input is oracle-distribution.
- GO being 16.98 means Stage-1 generated coarse alone causes damage, but PB1 can
  still remain roughly stable when C41/S4 are oracle.
- GG being 39.34 means generated Stage-1 plus generated Stage-1.5 conditions
  interact nonlinearly and break the downstream motion.
- GG pelvis drift of 33.91 cm is the most important qualitative signal. This is
  not merely hand contact missing. Root/world trajectory becomes unstable after
  Stage-1.5 is driven by generated Stage-1.

So the next experiment should target Stage-1.5's input distribution, not another
Stage-1-only loss ablation.

## 4. Be precise about what R41 did and did not refute

The reviewed document says R41 was "refuted" and that more Stage-1-only cascade
training is wasted GPU.

This is mostly directionally right, but the wording is too strong.

Better wording:

- R41 refuted the specific hypothesis that tuning Stage-1 alone, while using
  frozen PB1 and oracle C41/S4, can close the deployment gap.
- R41 did not refute Stage-1's importance.
- R41 did not refute joint Stage-1 plus Stage-1.5 training.
- R41 did not prove a universal hard GO ceiling. It measured the current
  Stage-1-only route under the current PB1 and condition setup.

Keep this nuance, otherwise the project may prematurely discard useful joint
cascade directions.

## 5. Recommended R43 implementation order

### Stage A: Add generated-cache conditioning to Stage-1.5 training

Modify `src/piano/training/train_stage1p5.py` surgically.

Goal:

- support `oracle`,
- support `generated_cache`,
- support `mixed`.

Minimum code shape:

1. Add config parsing for the new fields.
2. Make the dataloader/batch expose enough metadata to locate the cache file
   for each sample. If this metadata already exists, use it. If not, add the
   smallest dataset change needed.
3. Add a helper to load and validate generated `stage1_coarse`.
4. In the step function, select the Stage-1 cond source before building
   `cond["stage1_coarse"]`.
5. Keep the existing oracle behavior bit-identical when source is `oracle`.

Review after Stage A:

- Verify no generated-cache path is used as a normalizer.
- Verify missing cache fails hard.
- Verify shape, dtype, finite checks exist.
- Verify oracle mode still runs.
- Add at least one unit or smoke test for cache loading and mixed-source
  selection if the project test structure allows it.

### Stage B: Generate train+val Stage-1 cache for the chosen Stage-1 checkpoint

Use the existing substitute-condition sampling path:

`scripts/stage_a_generator/sample_substitute_conds_cli.py`

For the immediate R43 P0, use A2 unless the user explicitly decides to go back
to V8/V6.

Required buckets:

- train
- val

Before training, run a cache preflight:

- count expected clips vs found clips,
- key exists,
- shape `(T, 23)`,
- finite values,
- basic mean/std report against oracle,
- save a short `.md` and `.json` audit.

This can reuse the spirit of R35-style audits. The exact script can be new if
that is cleaner.

### Stage C: Train Stage-1.5 P0

Base recipe:

- R38-B1 architecture and losses.
- Target remains GT C41/S4.
- Input Stage-1 coarse comes from generated cache or a generated/oracle mixture.

Recommended first source policy:

- `mixed`, generated probability around 0.8.

Reason:

- pure generated-cache training may overfit to one Stage-1 checkpoint,
- pure oracle is the old failed distribution,
- mixture regularizes without hiding the generated-input problem.

This is not a numeric tuning request. It is a strategy-level guard against
over-specializing Stage-1.5 to one upstream checkpoint.

### Stage D: Rerun the R42 2x2 with the new Stage-1.5 checkpoint

Do not only run GG.

Run all four cells again:

- OO
- GO
- OG
- GG

Reason:

- GG tells us deployment improved.
- OG tells us whether Stage-1.5 lost oracle-distribution quality.
- OO and GO preserve comparability to the previous tarball.

### Stage E: Package the sync-back tarball

The package should include:

- the new R43 config(s),
- train log,
- validation log,
- cache audit `.md` and `.json`,
- R42-style 2x2 summaries,
- per-cell sustained/gait/body/G1 summaries and stats,
- a return document explaining what changed, what passed, and what remains
  uncertain.

Do not include huge `.npz` caches unless the pack script has an explicit opt-in.

## 6. P2 sigma augmentation: not fallback-only

The reviewed document treats sigma augmentation as fallback only.

I disagree.

Pure Gaussian sigma augmentation is not enough to model generated Stage-1's
structured errors, so it cannot replace generated-cache training. But once the
trainer can actually load generated cache, mild sigma augmentation is a useful
regularizer.

Recommended position:

- Do not run P2 as a replacement for P0.
- Consider including mild sigma augmentation inside P0, especially in mixed
  source mode.
- If P0 fails, diagnose why before treating sigma-only as the next serious
  candidate.

## 7. Success criteria

The proposed success bar is close, but make it comparative.

Primary success:

- new GG sustained drift mean should be near or below the old GO level,
  roughly `< 17-18 cm`;
- GG pelvis drift should drop from 33.91 cm to below 8 cm;
- GG track mean should recover above 0.85;
- GG low-track rate should fall clearly below the catastrophic 50.3%.

Regression checks:

- OG should not degrade badly from 11.94.
- Gait/body/G1 summaries should not show a new obvious failure.
- Hand drift should improve with pelvis, not simply move the error from pelvis
  to hands.

Do not declare victory from one scalar.

## 8. Concrete "do not do this" list

Do not:

- point `data.stage1_coarse_cache_root` at generated substitute caches and call
  that P0;
- silently fall back to oracle Stage-1 coarse when a generated cache entry is
  missing;
- write the R43 report as if R42 used V8/V6 unless you have rerun R42 with V8/V6;
- compare A2 and V8/V6 as if their generated Stage-1 distributions are
  interchangeable;
- run only GG after retraining;
- call R41 globally useless. It answered one question; it did not answer the
  joint-training question.

## 9. Return document requested after implementation

After you implement and run R43, write a return doc for Codex and the user.

Suggested path:

`analyses/2026-06-12_r43_return_for_codex.md`

It should include:

1. Exact commits used.
2. Exact Stage-1 checkpoint used for generated cache.
3. Whether the run used A2 or V8/V6, with evidence from logs.
4. Config diff summary.
5. Cache coverage and audit results.
6. Training outcome.
7. Full 2x2 metric table.
8. Whether GG improved without OG regression.
9. Known risks and next recommended experiment.

Please also perform a code review after each stage above before moving on to the
next one. If any review finds a blocking issue, stop and fix it before training.
