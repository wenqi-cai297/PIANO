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

Latest pushed commit should be checked with `git log -5 --oneline`; after the
v15/v16 local work, do not assume local `.md` files are fresher than git until
`git status` is checked.

## Read Order

Must read:

1. `PROGRESS.md` - current state, best numbers, active artifacts.
2. `PLAN.md` - next actions and routes that are no longer worth running.
3. `ANALYSIS.md` - compact index for durable analysis docs.
4. `analyses/stageB_compact.md` - consolidated Stage B evidence and decisions.
5. `analyses/2026-05-01_per_step_guidance_design.md` - v17 design + ablation.
6. `analyses/2026-05-01_v17_per_step_result.md` - v17-C single-sample SOTA
   result (matches v14 K=16 oracle on local error; per-step inner loop flips
   60.67% of base tokens). Load alongside the design doc.

Read when touching that area:

- `SPEC.md` - stable project design and code layout.
- `analyses/pseudo_label_pipeline.md` - Stage 1 labels and thresholds.
- `analyses/stageA_design.md` - Stage A predictor v6 shipped state.
- `analyses/early_setup.md` - one-time server/data/backbone gotchas.
- `scripts/README.md` - script layout convention.

Do not read old dated Stage B notes; they were merged into
`analyses/stageB_compact.md` on 2026-04-29.

## Current State, 2026-05-01

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

Latest evaluated branch:

- config: `configs/training/generator_v15_alignment_guided.yaml`
- runner: `scripts/stage_b_generator/run_v15_alignment_guided.sh`
- result: negative/neutral. `best_contact` raw full is `27.62 cm`,
  `full_guided` is `31.57 cm`, moving contact IoU is `0.3804`, moving correct
  GT-part recall is `0.1684`, and moving same-part local error is `55.09 cm`.
- local visualization: `runs/visualizations/stageB_v0_15_bc_review/{full,full_guided}`.

Latest evaluated branch (v16):

- config: `configs/training/generator_v16_alignment_mirror.yaml`
- runner: `scripts/stage_b_generator/run_v16_alignment_mirror.sh`
- data change: deterministic MoMask/HumanML3D-style mirror doubling for train
  only via `augmentation.mirror_duplicate=true`.
- validation/eval remain unaugmented.
- result: partial positive. `best_contact full` 26.79 cm (vs v15 27.62);
  `full_guided` 28.91 cm (vs v15 31.57). Best non-oracle correct GT-part
  recall to date: `0.1990` on `final` ckpt. Same-part local 52.91-53.49 cm
  (vs v15 54.24-59.92). Still 8-13 cm short of v14 K=16/K=64 oracle. Decision-
  gate verdict: keep mirror-doubling, but the next iteration must change
  the mechanism, not the data/loss knob.

Latest evaluated branch (v17-C):

- runner: `scripts/stage_b_generator/run_v17_per_step_guidance.sh`
- design: `analyses/2026-05-01_per_step_guidance_design.md`
- result: `analyses/2026-05-01_v17_per_step_result.md`
- mechanism: per-step decoded-geometric guidance — replaces the baseline
  MaskGIT loop with a re-rolled version that runs N AdamW inner steps on the
  predicted logits at each MaskGIT iteration before commit, using a
  relaxed-decode geometric loss with frozen baseline residuals
  (MaskControl ICCV 2025 `each_iter` half of the recipe; PIANO previously
  ran only the `iter_last` post-hoc half).
- inference-time only — runs on existing v14/v15/v16 `best_contact.pt`
  unchanged, no retraining.
- v17-C config: `PER_STEP_ITERS=10`, `GUIDANCE_STEPS=0` (per-step only).
- single-sample 80-clip result: contact `21.77 cm` / coupled `0.3428` /
  IoU `0.4388` / correct-part recall `0.2020` / same-part local `46.13 cm`.
  Same-part local matches v14 K=16 composite oracle (46.32 cm); coupling
  beats v14 K=64 alignment oracle (0.3339). Per-step inner loop flips
  60.67% of base tokens vs naive baseline.

Latest evaluated sweep (v17-D + v17-E, 2026-05-01):

- v17-D stacked (per_step=10 + post_hoc=30) is *worse* than v17-C → MaskControl's
  canonical stack does not stack on PIANO; do not pursue post-hoc.
- v17-E.20 (per_step=20 only): contact `18.62 cm`, correct-part `0.2639`.
- v17-E.50 (per_step=50 only): contact `16.50 cm` (below GT VQ roundtrip
  18.47 — metric-gaming red flag), correct-part `0.2746`. User visual review:
  visibly better than v16 raw, but contact patches still misaligned.

Latest follow-up findings (2026-05-01):

- D-A: γ_int ≈ 0.02 final after v14/v15/v16 training (zero-init grew to 0.02
  over 80 epochs). IntXAttn is heavily underused — architectural lever
  exists but deferred until v17-F decides whether inference TTT is enough.
- MaskControl source-verified: pretrained MoMask VQ + frozen base + pure
  CE training → **VQ codebook is not the bottleneck**. Codebook re-training
  deprioritised.

Latest evaluated sweep (v17-F Gumbel, 2026-05-01) — NEGATIVE:

- v17-F.10 / v17-F.20 (Gumbel ON) regress every metric vs Gumbel-OFF
  sanity reruns at both budgets. Root cause: PIANO's frozen baseline
  residual_emb_sum dominates the decode embedding magnitude, so Gumbel
  noise on the small base contribution destabilises inner-loop
  gradients. MaskControl ignores residual during per-iter so same
  injection works for them. Same multi-quantizer-residual
  incompatibility that killed v17-D.
- **Do not ship Gumbel on PIANO**. Default `--per-step-gumbel-scale=0.0`.
- Sanity reruns (v17-C-ng / v17-E.20-ng, Gumbel OFF) match originals
  within 0.5 cm — pipeline is consistent.

**Inference path now near-saturated**: post-hoc stacking and Gumbel
both regress; budget sweep at diminishing returns. Ship configs:
**v17-E.20** (contact 18.62, correct-part 0.264, local 42.09 cm) or
**v17-E.50** (contact 16.50, correct-part 0.275, local 39.02 cm; with
metric-gaming caveat per visual review).

Latest implemented branch, pending server results (v17-G γ_int boost):

- runner: `scripts/stage_b_generator/run_v17g_gamma_int_boost_sweep.sh`
- hypothesis: γ_int ≈ 0.02 final value (per D-A audit) is ~1/25 of
  typical ControlNet-style strength → IntXAttn cross-attention is
  underused → boost it at inference time and see if contact-patch
  misalignment improves.
- new CLI: `--gamma-int-boost FLOAT` (default 1.0 = no change). Boost
  is applied as in-place ×scale to all gamma_int / gamma_int_res
  parameters during the inference call, restored after.
- variants: v17-G.b{1,2,5,10,20} on top of v17-E.20 base config
  (per_step=20 Gumbel OFF).
- detail: `analyses/2026-05-01_v17f_gumbel_result_and_p1_plan.md`.

## Current Decision

Stop blind training-parameter sweeps for Stage B. v14 sampled-ST helps contact
and candidate-pool quality, but K=64 alignment oracle shows pure reranking is
near exhausted.

Next work:

1. Run v17-D + v17-E sweep on the server with the new wrapper:
   `bash scripts/stage_b_generator/run_v17_sweep.sh`. Three back-to-back
   conditions (v17-D stacked, v17-E.20, v17-E.50). Sync back
   `runs/eval/stageB_v0_17_v16bc_{stacked,per_step_iters20,per_step_iters50}_*`.
2. Compare each variant against v17-C baseline (contact `21.77`, coupled
   `0.3428`, IoU `0.4388`, correct-part `0.2020`, same-part local `46.13`).
   The remaining design threshold to clear is correct-part recall ≥ 0.22.
3. v17-D > v17-C on correct-part → ship v17-D. v17-E.20 ≈ v17-C → 10 iters
   saturated; drop v17-E.50. v17-E.50 > v17-E.20 by ≥ 1 cm contact and 1 pp
   correct-part → consider full 100. correct-part stays < 0.22 across all
   variants → pivot to OMOMO-style hand-position intermediate target.

Do not spend another main iteration on larger K, rerank-weight tuning, or
data-symmetry knobs.

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
- Stochastic mirror augmentation v0.7 (`mirror_prob=0.5`): mathematically
  correct but regressed contact. Deterministic mirror doubling is now separated
  as v16 because MoMask/HumanML3D train data is mirrored.
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
bash scripts/stage_b_generator/run_v16_alignment_mirror.sh
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
