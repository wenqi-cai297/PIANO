# 2026-05-03 — Unified metric results: training vs inference bottleneck diagnosis

Result of the 2026-05-02 metric overhaul. New ship-gate metrics (N1/N2
penetration, N3/N8 weighted local, N6 soft IoU, N7 jerk + KS distance)
implemented and run on all 22 v17 conditions + GT_orig + GT_roundtrip.

The unified table establishes the first end-to-end picture of where
PIANO sits along three axes (alignment, physical plausibility, motion
quality) and answers the user's question:

> **Is the bottleneck in training (learned distribution is flawed) or
> inference (sampling can't reach good samples)?**

**Short answer**: Training is the dominant bottleneck. Inference per-step
optimisation contributes ~50 % of available correct-part headroom but
hits a hard ceiling determined by the underlying raw distribution AND
trades alignment improvements for physical-plausibility degradation.

## 1. Reference numbers (from this measurement run)

```
GT_orig:          mean_min_dist 13.09 cm  | mean_pen 1.25 cm  | jerk 36 m/s^3
GT_roundtrip:     mean_min_dist 18.47 cm  | mean_pen 1.14 cm  | jerk 63 m/s^3

Codec floor on alignment metrics (GT_roundtrip vs GT_orig, N=80 clips):
  moving_contact_temporal_iou:   0.640
  moving_soft_contact_iou_pm2:   0.782
  moving_correct_part_recall:    0.393
  moving_weighted_local_error:   24.43 cm
  moving_weighted_target_error:  26.28 cm
```

## 2. Headline ship-gate table (all 22 v17 conditions + GT)

| condition                    | cont   | pen   | pen-2cm | IoU   | softIoU | corPt | %abs | wLoc  | jerk | KS-jerk |
|------------------------------|-------:|------:|--------:|------:|--------:|------:|-----:|------:|-----:|--------:|
| GT_orig                      | 13.09  | 1.25  |  41.2%  | —     | —       | —     | —    | —     |   36 | 0.000   |
| GT_roundtrip                 | 18.47  | 1.14  |  42.6%  | —     | —       | —     | —    | —     |   63 | 0.375   |
| v17-C v16bc                  | 21.77  | 1.51  |  44.5%  | 0.439 | 0.625   | 0.202 | 51%  | 29.95 |  283 | 0.343   |
| v17-D stacked                | 22.91  | 1.57  |  46.0%  | 0.438 | 0.621   | 0.196 | 50%  | 30.07 |  285 | 0.360   |
| v17-E.20 v16bc               | 18.62  | 1.67  |  50.0%  | 0.473 | 0.630   | 0.264 | 67%  | 29.36 |  282 | 0.355   |
| v17-E.50 v16bc               | 16.50⚠ | 1.71  |  55.0%  | 0.504 | 0.630   | 0.275 | 70%  | 29.98 |  290 | 0.394   |
| v17-F.10 Gumbel              | 23.53  | 1.31  |  41.6%  | 0.403 | 0.590   | 0.177 | 45%  | 29.39 |  280 | 0.338   |
| v17-F.20 Gumbel              | 19.36  | 1.58  |  48.2%  | 0.472 | 0.606   | 0.219 | 56%  | 29.54 |  281 | 0.336   |
| v17-G boost=1                | 18.67  | 1.65  |  50.3%  | 0.474 | 0.631   | 0.267 | 68%  | 29.57 |  283 | 0.357   |
| v17-G boost=2                | 19.93  | 1.69  |  52.9%  | 0.497 | 0.654   | 0.275 | 70%  | 30.42 |  286 | 0.378   |
| v17-G boost=5                | 82.32  | 0.70  |  30.7%  | 0.276 | 0.386   | 0.058 | 15%  | 31.92 |  302 | 0.489   |
| v17-G boost=10               | 110.58 | 0.37  |  20.7%  | 0.188 | 0.298   | 0.034 |  9%  | 32.35 |  307 | 0.564   |
| v17-G boost=20               | 109.78 | 0.30  |  18.2%  | 0.159 | 0.248   | 0.035 |  9%  | 32.41 |  309 | 0.561   |
| **B1: v17-E.20 final.pt**    | 19.69  | 1.53  |  49.7%  | 0.441 | 0.616   | 0.241 | 61%  | 28.93 |  285 | 0.350   |
| **B1: v17-E.50 final.pt**    | 16.86⚠ | 1.66  |  54.2%  | 0.507 | **0.656** | **0.292** | **74%** | **28.74** | 291 | 0.391 |
| B2 sanity (pm=0)             | 18.43  | 1.65  |  49.2%  | 0.476 | 0.651   | 0.265 | 68%  | 29.05 |  282 | 0.353   |
| B2 part_margin=0.5           | 18.73  | 1.59  |  50.1%  | 0.474 | 0.636   | 0.235 | 60%  | 29.84 |  283 | 0.356   |
| B2 part_margin=1.0           | 19.70  | 1.43  |  47.7%  | 0.470 | 0.638   | 0.198 | 50%  | 29.85 |  281 | 0.357   |
| B2 part_margin=2.0           | 21.17  | 1.52  |  46.7%  | 0.450 | 0.620   | 0.225 | 57%  | 30.13 |  283 | 0.363   |
| B2 pm=1.0 + sc=0.1           | 18.56  | 1.52  |  51.0%  | 0.492 | 0.648   | 0.222 | 57%  | 29.70 |  282 | 0.358   |
| B2 pm=1.0 + sc=0.5           | 18.39  | 1.60  |  51.7%  | 0.480 | 0.626   | 0.208 | 53%  | 29.24 |  282 | 0.355   |
| B2 pm=1.0 + sc=1.0           | 18.90  | 1.55  |  50.6%  | 0.497 | 0.643   | 0.233 | 59%  | 29.42 |  282 | 0.355   |

Units: cont/pen/wLoc in cm; jerk in m/s³; %abs = correct_part / codec_floor (= 0.393); KS-jerk = Kolmogorov-Smirnov distance to GT_orig jerk distribution.

⚠ = `mean_min_dist < codec floor 18.47 cm` — metric-gaming flag.

## 3. Key findings

### F-1. v17-E.50 + final.pt is the new project SOTA on alignment, but with confirmed metric gaming

- **correct-part recall 0.292** (74 % of codec ceiling absorbed) — project best
- **weighted local error 28.74 cm** — closest any model has come to codec floor 24.43 cm
- **soft IoU 0.656** — also project best
- BUT: `mean_min_dist 16.86 cm < codec floor 18.47 cm` (gaming flag #1)
- AND: `mean_pen 1.66 cm > GT_orig 1.25 cm` (gaming flag #2; +0.4 cm)
- AND: `frac_pen_gt_2cm 54.2%` vs GT_orig 41.2 % (+13 pp; gaming flag #3)
- AND: `mean_jerk 291 m/s³` vs GT_orig 36 m/s³ (**8 ×**; gaming flag #4)

The four independent flags converge: the alignment gains came at real
physical cost. Visual review (still pending) will likely confirm.

### F-2. All per-step inference variants share the same plausibility tax

All v17-* model conditions (excluding boost ≥ 5 catastrophes) have
mean_jerk in the 280–290 m/s³ range — **8 × GT_orig**, regardless of
budget (10/20/50 iters), Gumbel on/off, boost on/off, or part_margin
on/off. So **per-step inner-loop optimisation itself is the jerk source**,
not a particular configuration.

Penetration similarly: every per-step variant >= 1.5 cm mean (vs GT_orig
1.25), with v17-E.50 highest at 1.71. v17-C (10 iters, conservative) is
the lowest at 1.51 — still +0.26 vs GT_orig.

**Implication**: the per-step path has a hard physical-plausibility
ceiling. Trade-off curve is monotone — more iterations → better alignment
metrics → worse penetration / jerk.

### F-3. Codec floor is approached but never crossed on alignment

| metric                | best model    | codec floor | gap to floor |
|-----------------------|--------------:|------------:|-------------:|
| moving_correct_part   | 0.292 (E.50f) | 0.393       | **−0.101**   |
| weighted_local        | 28.74 (E.50f) | 24.43 cm    | **+4.31 cm** |
| soft_IoU_pm2          | 0.656 (E.50f) | 0.782       | **−0.126**   |

Even the best inference-side recipe sits clearly below codec ceiling on
all alignment metrics. **There IS still ~26 % correct-part headroom and
~4 cm weighted-local headroom available within the existing MoMask backbone.**
Whether that gap can be closed without paying further plausibility tax
is the open question.

### F-4. v17-G boost ≥ 5 has interpretable signature beyond "catastrophic"

The boost ≥ 5 conditions all show:
- `mean_pen` *drops* below GT_orig (0.30–0.70 cm vs 1.25 cm)
- `frac_pen_gt_2cm` drops to 18–31 % vs GT_orig 41 %
- `correct_part` drops to 0.034–0.058
- `mean_jerk` rises to 302–309 m/s³

The signature: motion is **flying off into space**, not even reaching the
object — penetration drops because the body is far away, contact recall
drops because the body isn't close, and jerk rises because the
inference-time recalibration of γ_int produces gibberish at the per-step
optimisation level.

### F-5. Soft IoU vs hard IoU gap reveals timing-tolerance signal

| condition         | hard IoU | soft IoU pm2 | gap |
|-------------------|---------:|-------------:|----:|
| GT_roundtrip      | 0.640    | 0.782        | +14 pp |
| v17-E.50 final.pt | 0.507    | 0.656        | +15 pp |
| v17-E.20 final.pt | 0.441    | 0.616        | +18 pp |

Soft IoU consistently +14–18 pp over hard IoU. **Roughly 15 % of "missed
contact frames" are timing-mismatched (within ±2 frames, i.e. ±0.1 s)
rather than completely missed.** This isn't a model failure — it's a
metric strictness artifact. Going forward, soft IoU is more informative
as ship gate; hard IoU is the rigid timing diagnostic.

### F-6. B2 part_margin / segment_consistency NEGATIVE confirmed in unified table

Even on the new metric set, every B2 variant (pm > 0) regresses
correct-part recall vs sanity (pm=0):

| variant          | corPt  | %abs | wLoc  | wTgt  |
|------------------|-------:|-----:|------:|------:|
| sanity (pm=0)    | 0.265  | 68%  | 29.05 | 29.21 |
| pm=0.5           | 0.235  | 60%  | 29.84 | 29.76 |
| pm=1.0           | 0.198  | 50%  | 29.85 | 29.81 |
| pm=2.0           | 0.225  | 57%  | 30.13 | 30.10 |

Decisive: training-time aux terms do not transfer to inference per-step
in a positive way. Mechanism (residual drift amplification, established
in the 2026-05-02 B3 result) is consistent across all metrics.

## 4. Diagnosis: training vs inference

### Decomposition of correct-part gap (the most discriminating metric)

Reference points:
- raw v16 best_contact correct-part: ≈ 0.176 (PROGRESS, prior measurement)
- raw v16 final.pt correct-part: 0.199 (PROGRESS, prior measurement)
- guided v17-E.50 + best_contact: 0.275 (this run)
- guided v17-E.50 + final.pt: 0.292 (this run)
- codec floor: 0.393

Decomposition:
```
training_contribution_raw           = 0.199 − 0.176 = +0.023 (10.6 % of raw→floor gap)
inference_contribution_on_bc        = 0.275 − 0.176 = +0.099 (45.6 % of raw→floor gap)
inference_contribution_on_final     = 0.292 − 0.199 = +0.093 (47.9 % of raw→floor gap)
remaining_gap_from_E50_final        = 0.393 − 0.292 = +0.101 (52.1 % of raw→floor gap still uncaptured)
```

**Interpretations**:

1. **inference path has captured ~half of available headroom** —
   substantial, but bounded. Going from raw 0.199 → 0.292 is a 47 %
   absorption of the residual gap to codec floor.
2. **training improvements (best_contact → final.pt) translate ~1:1
   into guided gains** — +0.023 raw → +0.017 guided. So 1 pp of
   raw improvement ≈ 1 pp of guided improvement. **training-side
   investment yields the same magnitude on the ship metric as
   inference-side investment, with the added benefit of NOT paying
   plausibility tax.**
3. **52 % of correct-part headroom is still uncaptured** — and the
   inference path is hitting a plausibility ceiling (jerk 8 × GT, pen +0.4 cm).
   Closing the remaining gap with inference alone would amplify the
   plausibility tax further; a training-side intervention that lifts raw
   correct-part by another 5–10 pp would be the highest ROI move.

### Bottleneck classification

| component              | dominant bottleneck? | evidence |
|------------------------|----------------------|----------|
| Raw distribution quality | **YES (primary)** | raw correct-part 0.199 ≪ codec floor 0.393; even perfect inference can't fully close from this baseline; physical-plausibility tax limits how much inference can push |
| Inference sampling     | partially | per-step has captured 47 % of available headroom; further pushes hit jerk/penetration ceiling |
| VQ codec               | secondary | floor is 0.393 (60 % of perfect), but model is 0.292 — closer model to floor first, then revisit codec |
| γ_int gate calibration | minor / OOD-fragile | v17-G boost flat below 2x, catastrophic above; not the main lever |

**Plain-language version**: The model has **learned a contact distribution
that's missing the right modes**. Inference per-step is good at
*selecting* aligned commit decisions from what's available, but it can't
manufacture modes that aren't there — it can only push base tokens
toward locally-better positions, and it pays jerk + penetration tax
for that. The raw distribution itself needs more aligned-mode coverage.

## 5. Updated decision tree

| branch | category | priority | rationale |
|--------|----------|----------|-----------|
| **N1** visual review of v17-E.50 + final.pt | gate | **block** | Four independent metric-gaming flags (cont, pen, pen-frac, jerk). Strong signal it shouldn't ship; visual confirms or refutes. Predicted: visual flags wrong-patch failure. |
| Ship default change → **v17-E.20 + final.pt** | ship | high | wLoc 28.93, corPt 0.241 (61 % absorbed), pen 1.53, jerk 285. mean_min_dist 19.69 > codec floor (no gaming). Defensible, slightly behind v17-E.20 + best_contact on raw alignment but better on weighted_local. |
| **N2 = mid-loop residual refresh (B3')** | inference | medium | Predicted upside narrowed: 1–3 pp correct-part. Hits same plausibility ceiling. Worth doing as a final inference-side optimisation but not the major lever. |
| **B6 alignment-aware VQ retrain** | training | **highest** | Codec floor 0.393 is dominant alignment ceiling (60 pp lost to codec alone in the perfect-vs-floor gap). Retraining VQ-VAE specifically for contact-relevant joints (hands/feet/pelvis) could raise the ceiling materially. Cost: significant (1+ week). Justification grew stronger with this measurement. |
| **B4 = P2 γ_init re-init + finetune** | training | high | Test whether trained network can absorb a moderate γ_int change (0.05/0.1/0.2 candidates, NOT 0.5/1.0). This is the cheapest training-side experiment — 1 day code, 1 day server. Lower risk than B6 because no architecture change. |
| **B7 OMOMO-style explicit contact_target as input** | training | medium | Architecture change. Use Stage A predictor's already-computed `contact_target_xyz` as an input channel. Decouples from γ_int gate entirely. Cost: 1 week. Justified if B4 doesn't move the needle. |

### Recommended sequence

1. **N1** (visual review) → confirm v17-E.50 + final.pt is gamed (predicted) → ship default → **v17-E.20 + final.pt** (defensible, code-free switch).
2. **N2** (B3' mid-loop residual refresh) — implement and measure. If it adds 2+ pp correct-part: ship update. Otherwise close inference-side path.
3. **B4** (P2 γ_init = {0.05, 0.1, 0.2}) — first training-side experiment in 6 weeks. Cheapest training intervention.
4. If B4 not enough: **B6** (alignment-aware VQ retrain) — biggest expected upside but biggest investment.
5. Long-term backup: **B7** (architecture change with explicit contact_target input).

## 6. Metric set finalisation

Based on this measurement run, the recommended ship-gate set is:

**Ship gates (decision-relevant, must report)**:
1. `mean_min_dist` (cm) — **must be ≥ codec floor 18.47 cm; below = gaming**
2. `mean_part_penetration` (cm) — vs GT_orig 1.25 baseline
3. `frac_frames_pen_gt_2cm` — vs GT_orig 41.2 %
4. `moving_correct_part_recall` + `% of codec floor 0.393 absorbed`
5. `moving_weighted_local_error` (cm) + `delta vs codec floor 24.43 cm`

**Auxiliary (look but don't gate)**:
- `moving_contact_temporal_iou` (rigid)
- `moving_soft_contact_iou_pm2` (timing-robust)
- `moving_coupled_frame_frac` (binary)
- `moving_mean_best_kin_score` (continuous coupling)
- `mean_jerk` + `KS_distance_to_GT_jerk_distribution`

**Removed / superseded**:
- `same_gt_part_local_position_error` (subset-only mean — superseded by weighted_local)

## 7. Sources

- Reproducer: `python scripts/stage_b_generator/summarize_unified_metrics.py` (table + JSON dump)
- Per-condition penetration: `runs/eval/_unified_metrics/penetration/<label>_summary.json`
- Per-condition jerk samples: `runs/eval/_unified_metrics/jerk/<label>_samples.npz`
- Codec floor measurement: `runs/eval/stageB_codec_floor_alignment/summary.json`
- Unified summary: `runs/v17h_unified_summary.json`
- Implementation:
  - `scripts/stage_b_generator/measure_penetration.py` (N1/N2)
  - `scripts/stage_b_generator/measure_motion_quality.py` (N7)
  - `scripts/stage_b_generator/measure_contact_alignment.py` (added N3/N6/N8 fields)
  - `scripts/stage_b_generator/run_unified_metrics_eval.sh` (batch)
  - `scripts/stage_b_generator/run_realign_all.sh` (re-run alignment with new fields)
- Prior:
  - `analyses/2026-05-02_codec_floor_baselines.md` (codec floor finding)
  - `analyses/2026-05-02_v17h_results.md` (B1+B2+B3 server results)
  - `analyses/2026-05-01_v17_re_diagnosis.md` (re-diagnosis driving B1/B2/B3)
