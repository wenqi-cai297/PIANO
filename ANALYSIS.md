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
| `analyses/2026-05-01_v17_diagnostics_and_gumbel.md` | v17 follow-up — γ_int audit (final ≈ 0.02, IntXAttn heavily underused) + MaskControl source diff verification (VQ codebook is not the bottleneck) + Gumbel-Softmax relaxation added to per-step inner loop (matches MaskControl `each_iter`). |
| `analyses/2026-05-01_v17f_gumbel_result_and_p1_plan.md` | v17-F result (Gumbel **negative** on PIANO — regresses every metric at both budgets; multi-quantizer residual incompatibility) + P1 γ_int inference-boost plan (v17-G sweep at boost ∈ {1, 2, 5, 10, 20}). Inference path near-saturated; remaining lever is architectural γ_int gate. |
| `analyses/2026-05-01_v17g_gamma_int_boost_result.md` | v17-G result (γ_int boost-at-inference **negative** — boost ≥ 5 catastrophic, boost = 2 mixed) + close-out summary of v17 inference-side series + P2 plan (re-init γ_int + finetune Stage B; first training-side experiment after 6 weeks of inference iteration). |
| `analyses/2026-05-01_v17_re_diagnosis.md` | **Source-level re-diagnosis of the v17 series.** Refines (does not refute) the "γ_int undertrained" diagnosis. Surfaces 2 un-tested inference-side levers: (1) `final.pt` was never tested with v17-E (correct-part 0.199 vs best_contact 0.176); (2) per-step inner loss is a strict subset of training loss (no `part_margin` or `segment_consistency`). Revised decision tree B1–B5 with B1+B2 cheaper than P2. |
| `analyses/2026-05-02_v17h_results.md` | **B1+B2+B3 server results.** B1 v17-E.50 + final.pt → project SOTA (correct-part 0.292, local 36.11 cm). B2 part_margin / segment_consistency NEGATIVE on PIANO. B3 residual drift mean 6–12 cm explains B2 failure (drift scales with part_margin weight). New ship config `v17-E.50 + final.pt` pending visual review. Next branch: N2 = mid-loop residual refresh `--per-step-residual-refresh-every`. |
| `analyses/2026-05-02_codec_floor_baselines.md` | **VQ codec floor on alignment metrics — paradigm shift.** Previously-unmeasured: GT_roundtrip vs GT_orig codec floor on moving correct-part recall is **0.393** (not 1.0); same-part local **28.61 cm**; IoU **0.640**. v17-E.50+final.pt has absorbed 74% of inference-side correct-part headroom. v17-E.50 mean_min_dist 16.86 < codec floor 18.47 = direct metric-gaming evidence. **VQ codec is the dominant remaining bottleneck on alignment metrics** (contradicts prior claim based on mean_min_dist). Ship default switched to v17-E.20+final.pt pending penetration metric. New possible branch B6: alignment-aware VQ retrain. |
| `analyses/2026-05-03_gamma_int_re_evaluation.md` | **γ_int re-evaluation — supersedes prior "1/25 of ControlNet" framing.** Full v4–v16 trajectory: all converge to 0.017–0.036; v05 (160 epochs) reaches 0.036, slow growth past 80 epochs (extrapolated γ=0.05 ≈ 480 epochs). z_int contribution at γ=0.02 measured: correct-part +6.6 pp vs text_only (23 % of z_int headroom captured). ControlNet is NOT directly comparable (no scalar γ); LLaMA-Adapter is the right anchor, ratio holds for architectural reasons (gradient path × 8 layers, 100 × less data, lower-rank conditioning). Realistic P2 upside revised: +3–6 pp correct-part. P2 candidates revised to {0.05, 0.10, 0.20}; 0.5/1.0 EXCLUDED. |
| `analyses/2026-05-03_pseudo_label_v12_strict_design.md` | **v12 strict pseudo-label design (r3, two-case OR loose-distance).** Replaces v11's "approach within 12 cm" with "real contact" criteria matching OMOMO/CHOIS/InterDiff convention. PC-eval frame frac r3 45% vs v11 78% (neuraldome 25% → 36%, chairs 26% → 73%, imhd 19% → 33%, omomo 7% → 38%). User reviewed two false-negative reports (racket+glove case → r2 OR; sit at 16 cm pelvis dist → r3 loose). 24/27 r2-dropped clips recovered under r3. Server runner + verify script landed (commit `e16a59d`). |
| `analyses/2026-05-04_predictor_v7_target_diagnosis.md` | **RETRACTED 2026-05-05.** "v6 baseline 5-10 cm" column was fabricated; actual v6 L2 is 21.13 cm. See 2026-05-05 doc for correction. |
| `analyses/2026-05-05_v7fix_results_and_v6_baseline_correction.md` | **v7-fix accepted; v6 baseline correction.** Same eval set: v6 21.13 cm, v7 21.66 cm, v7-fix 21.77 cm — 21 cm is the architecture's normal performance, not a regression. v7-fix improvements: contact macro_f1 +22 % rel. (0.195 → 0.237), target L2 ~unchanged. **Stage B v18 unblocked**; ship v7-fix as Stage A predictor for v12-strict labels. v7-fix wandb csv synced from server is bit-identical to v7's (re-export error); eval JSONs are genuine. v8 backlog: representation change for target xyz (pelvis-relative or object-anchored) — not a v7-fix follow-up. |
| `analyses/2026-05-03_unified_metric_results.md` | **Unified metric overhaul + training-vs-inference diagnosis.** New ship gates (N1/N2 penetration, N3 weighted_local, N6 soft IoU, N7 jerk + KS) measured on GT_orig + GT_roundtrip + 22 v17 conditions. v17-E.50+final.pt has 4 independent metric-gaming flags (mean_min_dist < codec floor; penetration +0.4 cm; pen-2cm frac +13 pp; jerk **8 × GT_orig**). **Training is the dominant bottleneck** (52% of correct-part headroom uncaptured by best inference; per-step pays jerk×8 plausibility tax). Ship default → v17-E.20+final.pt; next training-side branch is B4 (γ_init ∈ {0.05, 0.1, 0.2}) followed by B6 (codec retrain). |

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
