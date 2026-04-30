# PIANO Analysis Index

This is the compact map of durable analysis docs. Dated Stage B logs were
merged on 2026-04-29 to reduce context load.

## Must-Read Docs

| File | Purpose |
|---|---|
| `analyses/stageB_compact.md` | Consolidated Stage B design, literature/code evidence, result timeline, negative results, and next diagnostics. |
| `analyses/stageA_design.md` | Stage A Interaction Predictor v6 shipped state and revisit triggers. |
| `analyses/pseudo_label_pipeline.md` | Current pseudo-label fields, thresholds, abandoned label paths, and known label risks. |
| `analyses/early_setup.md` | Server/data/backbone setup facts that are easy to forget. |
| `analyses/2026-05-01_per_step_guidance_design.md` | v17 per-step decoded-geometric guidance design — MaskControl source-code refs, PIANO-specific multi-quantizer adaptation, hyperparameter starting points, ablation matrix v17-A..E with decision rule, risk register. v17-C done; v17-D/E pending. |
| `analyses/2026-05-01_v17_per_step_result.md` | v17-C result — single-sample SOTA on v16 best_contact ckpt. contact 21.77 cm; same-part local 46.13 cm matches v14 K=16 composite oracle; moving_coupled 0.3428 beats v14 K=64 alignment oracle. Per-step trace shows 60.67% base-token flip rate. |

Root memory docs:

| File | Purpose |
|---|---|
| `restart_prompt.md` | Fast recovery checklist for fresh sessions. |
| `PROGRESS.md` | Current numbers and artifact state. |
| `PLAN.md` | Next actions and routes not worth repeating. |
| `SPEC.md` | Stable compact project specification. |
| `SUGGESTION.md` | Current recommendation memo. |

## Current Stage B Conclusion

v12 weight sweep is a negative result for more parameter tuning: stronger
decoded-contact surrogate and larger gradient share did not improve generated
contact. v13 target-trajectory loss also stayed near the same hard-sampling
line (`31.57 cm`, moving coupled `0.265`).

v14 sampled-ST is a positive result: `best_contact` reaches `27.37 cm` full
contact on the matched 80-clip eval, versus GT roundtrip `18.47 cm`. More
importantly, v14 K=16 reaches GT-roundtrip contact and improves coupling:
distance oracle `16.80 cm`; composite oracle `17.17 cm`, saved-best remeasure
`17.94 cm`, moving coupled `0.3715`.

Visual review and the new contact-alignment diagnostic show that this is still
not GT-quality generation. v14 K=16 composite reaches the distance band but has
only `0.4472` moving contact IoU against GT roundtrip, only `0.2378` correct
GT-body-part recall on moving GT-contact frames, and about `46 cm` same-part
object-local position error. The remaining bottleneck is now more specific
than "sample selection": the metric/guidance must be body-part and
contact-target aware.

The v14 K=64 alignment-aware oracle is a useful negative result. It improves
same-part local position error to `40.30 cm`, but contact remeasure is
`18.71 cm`, moving-coupled frame fraction falls to `0.3339`, moving contact IoU
is only `0.4516`, and correct GT-part recall is only `0.2496`. Across the full
K=64 pool, the best primary alignment error per clip is still `37.0 cm` on
average and the best moving same-part recall is only `0.165`. So this is not
just an underpowered K=16 reranking issue; v14 usually does not contain a truly
GT-aligned manipulation sample to select.

Latest result and implementation:

- v15 has been evaluated and is negative/neutral: `best_contact` raw full is
  `27.62 cm`, moving contact IoU is `0.3804`, moving correct GT-part recall is
  `0.1684`, and same-part local error is `55.09 cm`. `full_guided` worsens
  contact to `31.57 cm`.
- v15's code remains as the alignment-loss baseline: wrong-part margin,
  contact-segment consistency, strict contact-eval alignment metrics, and
  `--guidance-layers full_rvq`.
- v16 is now the active next implementation in
  `configs/training/generator_v16_alignment_mirror.yaml` and
  `scripts/stage_b_generator/run_v16_alignment_mirror.sh`: it keeps v15's
  objective and enables deterministic original+mirror training-set doubling.

Next analysis should be:

1. Run v16 on the server and compare raw `full` vs guided `full_guided`, using
   predicted/conditioned contact body part, object-local target, local-frame
   coupling, and the mirror-doubled training distribution.
2. Evaluate with contact distance, temporal coupling, and
   `scripts/stage_b_generator/measure_contact_alignment.py`.
3. Subset/hard-case review, especially IMHD and NeuralDome moving-object
   failures that remain above `25-40 cm`.

## Analysis Hygiene

When a new experiment finishes:

1. Record only durable facts: config, checkpoint, eval set, metric table,
   interpretation, and decision.
2. Update `PROGRESS.md` and `PLAN.md`.
3. If the result changes Stage B direction, append or revise
   `analyses/stageB_compact.md`.
4. Avoid creating long dated notes unless the result cannot yet be merged.
5. Delete or merge dated notes once the decision has settled.

Keep citations compact. Prefer full details in code comments, configs, and
artifact paths rather than pasting long source excerpts.
