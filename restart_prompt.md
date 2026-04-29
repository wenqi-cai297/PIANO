# Restart Orientation

Purpose: restore enough PIANO project context after a fresh session to make
correct decisions without loading the old experiment logs. This file is a
pointer and checklist, not a full lab notebook.

## First Commands

From repo root:

```bash
git status --short --branch
git log -5 --oneline
```

Latest pushed commit should be checked with `git log -5 --oneline`; before the
v15 implementation work, the remembered pushed commit was `871d35d Add Stage B
contact alignment diagnostics`.

## Read Order

Must read:

1. `PROGRESS.md` - current state, best numbers, active artifacts.
2. `PLAN.md` - next actions and routes that are no longer worth running.
3. `ANALYSIS.md` - compact index for durable analysis docs.
4. `analyses/stageB_compact.md` - consolidated Stage B evidence and decisions.

Read when touching that area:

- `SPEC.md` - stable project design and code layout.
- `analyses/pseudo_label_pipeline.md` - Stage 1 labels and thresholds.
- `analyses/stageA_design.md` - Stage A predictor v6 shipped state.
- `analyses/early_setup.md` - one-time server/data/backbone gotchas.
- `scripts/README.md` - script layout convention.

Do not read old dated Stage B notes; they were merged into
`analyses/stageB_compact.md` on 2026-04-29.

## Current State, 2026-04-30

Project goal: PIANO generates object-adaptive human motion by inserting a
structured interaction latent `z_int` between text/object inputs and the
motion generator.

Stage 1 pseudo-labels: current InterAct pseudo-label track is v11.

Stage A predictor: shipped v6. Predictor of record is the server checkpoint
identified as `runs/training/predictor/final.pt`; local sync may contain only
its eval JSONs, not the `.pt` file.

Stage B generator: active bottleneck. Current best evaluated one-shot contact
checkpoint after v14 sampled-ST training is:

- server checkpoint identifier:
  `runs/training/generator_v14_sampled_st_contact/best_contact.pt`
- full contact on matched 80-clip eval: `27.37 cm`
- text_only: `57.82 cm`
- swap: `74.79 cm`
- moving-object coupled frame fraction: `0.2765`
- v14 K=16 distance oracle: `16.80 cm`
- v14 K=16 composite oracle: `17.17 cm` oracle, `17.94 cm` saved-best
  remeasure, moving-coupled `0.3715`
- v14 K=16 composite contact alignment to GT roundtrip: moving contact IoU
  `0.4472`, moving correct GT-part recall `0.2378`, same-part object-local
  position error `46.32 cm`
- v14 K=64 alignment oracle: `17.92 cm` oracle, `18.71 cm` saved-best
  remeasure, moving-coupled `0.3339`, moving contact IoU `0.4516`, correct
  GT-part recall `0.2496`, same-part object-local position error `40.30 cm`
- GT original: `13.09 cm`
- GT VQ roundtrip: `18.47 cm`

Interpretation: `z_int` is active and v14 is the first clear single-sample
spatial-contact gain after the v12/v13 plateau. The v14 K=16 candidate pool is
also better than v12 K=16 and reaches GT-roundtrip contact with better coupling.
Visual review and contact alignment show the selected samples are still not
GT-quality: distance/composite can be good while the wrong body part, wrong
phase, or wrong object-local patch is used. K=64 alignment-aware selection is
not enough: it improves local position error modestly but worsens coupling, and
the best available K=64 candidates still have `37.0 cm` mean primary alignment
error and only `0.165` best moving same-part recall. The remaining problem is
the generated distribution's lack of aligned manipulation samples, not just a
reranker weakness.

Latest implemented branch, pending server results:

- config: `configs/training/generator_v15_alignment_guided.yaml`
- runner: `scripts/stage_b_generator/run_v15_alignment_guided.sh`
- loss change: wrong-part margin + contact-segment consistency on the decoded
  target-trajectory objective.
- monitoring change: contact eval logs strict alignment metrics and selects
  `best_contact.pt` by `alignment_contact_score`.
- guidance change: `qual_eval.py --guidance-layers full_rvq` optimizes the full
  generated RVQ stack in decoded target space and writes `full_guided`.

## Current Decision

Stop blind training-parameter sweeps for Stage B. v14 sampled-ST helps contact
and candidate-pool quality, but K=64 alignment oracle shows pure reranking is
near exhausted.

Next work:

1. Run v15 on the server, then sync the listed train/eval/wandb outputs back.
2. Optimize predicted contact body part, object-local `contact_target_xyz`, and
   local-frame coupling together, not only any-part min-distance.
3. Beat both v14 K=16 composite (`17.94 cm`, coupled `0.3715`, moving IoU
   `0.4472`, correct-part recall `0.2378`) and v14 K=64 alignment
   (`18.71 cm`, coupled `0.3339`, moving IoU `0.4516`, correct-part recall
   `0.2496`, local error `40.30 cm`).

Do not spend another main iteration on larger K or rerank-weight tuning alone.

## Stage B Routes Already Tested

Meaningful gains:

- v0.4: fixed MoMask encoder normalization; solved token/body collapse.
- B0: added contact distance metric; changed the objective from subjective
  mp4 viewing to a measured body-object distance.
- v0.6: per-head gamma improved the canonical 5-clip contact result.
- v0.9: decoded contact auxiliary loss made `z_int` matter for contact.
- v0.10: full-RVQ decoded-contact path helped on 20 clips, but did not close
  the gap.
- v14: sampled-ST full-RVQ decoded auxiliary path improved one-shot contact to
  `27.37 cm`; v14 K=16 composite reaches `17.94 cm` with coupled `0.3715`.
- v14 K=64 alignment oracle: improves same-part local error to `40.30 cm`, but
  contact/coupling do not improve and the candidate pool is still poorly
  aligned.

Negative or exhausted routes:

- More CE training: lowered CE but worsened contact.
- Mirror augmentation: mathematically correct but regressed contact.
- Trainable-copy InterControl variant: fixed dead init, still regressed.
- B3 inference-time base-logit guidance: produced mixed wins/losses; base
  logits are not a stable binding lever.
- C1 residual `z_int` wiring alone: failed.
- C2b decoded-contact weight sweep: surrogate improved; generated contact did
  not.

## Operational Notes

Server repo path used in previous runs:

```bash
cd /media/gpu-server-1/4TB_for_data/Cai/PIANO/PIANO
conda activate piano
```

Current train/eval runner:

```bash
bash scripts/stage_b_generator/run_v15_alignment_guided.sh
```

Local Windows workspace:

```text
e:\Project\2026-04-13
```

Most root memory docs and `analyses/` are gitignored local workflow files. If
the user asks to commit them, force-add explicitly.

## Before Acting

Read files before answering repo-specific questions. Prefer official/upstream
functions for MoMask, SMPL, VQ decode, rotations, and motion recovery; do not
reimplement nontrivial math unless no upstream path exists. Keep Python
src-layout: library code in `src/piano/`, direct scripts in `scripts/`.
