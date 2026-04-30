# 2026-05-01 — v17-G γ_int inference-boost result (negative) + P2 plan

Closes the v17 inference-time TTT series. v17-G is **negative**:
γ_int boost ≥ 5 catastrophic; boost = 2 mixed/neutral. The lever
exists architecturally (D-A audit + `swap` column from this run both
confirm γ_int controls how much z_int reaches the base path) but
**cannot be applied at inference time** — the rest of the trained
MaskTransformer is calibrated to the γ_int ≈ 0.02 it was trained
under, and can't tolerate an inference-time scale change.

The next branch is P2 (re-init γ_int with positive constant + finetune
Stage B from v16 ckpt). This is the first **training-side** experiment
after 6 weeks of inference-side work.

## v17-G result

Sweep on v16 best_contact ckpt + v17-E.20 base config (per_step=20,
Gumbel OFF, full_rvq post-hoc=0). 80-clip matched eval.

| variant | raw cont | per-step cont | raw IoU | per-step IoU | raw correct | per-step correct | raw local | per-step local |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v17-E.20-ng (no boost, prior rerun) | 26.79 | 18.19 | 0.382 | 0.475 | 0.176 | 0.271 | 53.49 | 41.90 |
| **v17-G.b1**  (boost=1, sanity) | **26.79** | 18.67 | 0.382 | 0.474 | 0.176 | 0.267 | 53.49 | 42.31 |
| **v17-G.b2**  (boost=2)         | 25.99 | 19.93 | **0.425** | 0.497 | **0.151** | 0.275 | **60.72** | **46.99** |
| **v17-G.b5**  (boost=5)         | **126.20** | **82.32** | 0.202 | 0.276 | 0.047 | 0.058 | 186.11 | 136.48 |
| **v17-G.b10** (boost=10)        | 167.83 | 110.58 | 0.115 | 0.188 | 0.016 | 0.034 | 240.05 | 169.92 |
| **v17-G.b20** (boost=20)        | 164.03 | 109.78 | 0.106 | 0.159 | 0.020 | 0.035 | 236.55 | 169.03 |

Sanity (b1 vs prior v17-E.20-ng rerun): raw 26.79 identical, per-step
contact 18.67 vs 18.19 within RNG drift, all other metrics within
noise. ✓ Pipeline consistent.

### z_int conditioning sanity (text_only / swap)

| boost | text_only | swap |
|---:|---:|---:|
| 1  | 53.75 | 69.67 |
| 2  | 53.75 | 97.73 |
| 5  | 53.75 | 197.88 |
| 10 | 53.75 | 227.86 |
| 20 | 53.75 | 230.85 |

- **text_only invariant** at 53.75 across all boosts → z_int is null
  in this condition, boost has nothing to amplify. Confirms boost
  mechanism doesn't accidentally modify text-only path.
- **swap monotonically blows up** with boost → wrong z_int amplified
  by boost makes motion deviate further. Confirms γ_int boost is
  actually controlling z_int's effect strength.

So: the mechanism IS plumbed correctly. The catastrophe at boost ≥ 5
is downstream — the trained network can't absorb the larger gate.

### boost = 2 mixed-signal interpretation

- raw `full` IoU: 0.425 vs 0.382 baseline (+4.3 pp) — **measurable
  increase in z_int's effect on raw generation**
- raw `full` correct-part: 0.151 vs 0.176 (−2.5 pp) — wrong-part
  contact slightly increases too (more z_int = more aggressive contact
  pursuit but not necessarily on the right body part)
- raw `full` local: 60.72 vs 53.49 (+7.2 cm) — body parts move toward
  contact targets but in wrong patches
- per-step `full_guided`: every metric slightly worse (−0.5 to +5 cm
  contact, IoU + a bit, correct-part flat, local +5 cm)

Boost = 2 is the only non-catastrophic non-trivial point. The mixed
signal ("z_int has more effect on raw, but per-step doesn't compound")
suggests γ_int boost shifts the distribution but doesn't actually
unlock new capability — the network's z_int interpretation logic was
calibrated for γ ≈ 0.02 and a 2x scaling already starts producing
incorrect contact-part decisions.

## Diagnosis: γ_int is undertrained, not under-applied

D-A had hypothesised "IntXAttn is gated nearly shut → boost should
help". v17-G refutes the boost-at-inference part. Refined hypothesis:

**γ_int finished training around 0.02 because that's where the joint
optimum sat for the v9–v16 training objective**. The CE + decoded
contact aux loss didn't push γ_int higher because:

1. The decoded contact aux loss is computed in decoded motion space,
   far downstream of γ_int. Backprop through 8 transformer layers +
   relaxed VQ decode + ric recovery + L2 dilutes γ_int's gradient
   signal.
2. CE on tokens is largely satisfied by text + object pose channels;
   z_int's contribution is marginal at the token-prediction objective.
3. Zero-init for γ_int means it starts at zero gradient (because
   IntXAttn output * 0 = 0). Only loss-induced perturbations from
   neighbouring layers can grow it. The growth rate is necessarily
   slow.

To make γ_int grow during training we need either:
- a stronger gradient signal at the IntXAttn output (longer training
  doesn't help — v14/v15/v16 all hit ~0.02 plateau by epoch 30-40 per
  the wandb csv),
- a different init that starts higher (P2),
- a different loss that rewards z_int usage directly (e.g., per-layer
  contact-target MSE on IntXAttn output projected to motion space —
  much more invasive).

**P2 is the cheapest test**: re-init γ_int to a positive constant
before finetune; let the network adjust its other layers' calibration
to absorb a larger gate.

## v17 series summary (close out the inference-side branch)

| version | intervention | result | ship status |
|---|---|---|---|
| v17-C    | per-step iters=10, Gumbel OFF, no post-hoc | contact 21.77 / correct 0.20 | ship-able |
| v17-D    | per-step + post-hoc stack (canonical MaskControl) | **negative** — post-hoc hurts on PIANO's deep RVQ | DO NOT ship |
| v17-E.20 | per-step iters=20 | contact 18.62 / correct 0.26 | **current ship recommended** |
| v17-E.50 | per-step iters=50 | contact 16.50 (< GT roundtrip ⚠) / correct 0.28 | metric-gaming risk; ship only with visual QA |
| v17-F    | + Gumbel-Softmax (matches MaskControl) | **negative** — multi-RVQ residual incompatibility | DO NOT ship |
| v17-G    | + γ_int inference boost | **negative** — boost ≥ 5 catastrophic, boost = 2 mixed | DO NOT ship |

**Inference-time TTT path is now SATURATED on PIANO**. Five distinct
levers (per-step budget, post-hoc stacking, Gumbel noise, residual
context, γ_int gate) tested. Three negative, one mixed (post-hoc
guidance from v15/v16 helped slightly when alone, hurt when stacked
with per-step), one positive (per-step itself, the v17-C/E ladder).
Diminishing returns clear.

The remaining gap to the desired contact quality (visual review:
correct broad area but wrong patch) is structural — needs training-
side work.

## P2 — re-init γ_int + Stage B finetune (next branch)

### Hypothesis

If γ_int's small final value was an optimisation accident (zero-init
decayed gradient + loss objective that doesn't reward γ growth) rather
than a structural property of the architecture, then **re-initing
γ_int at a positive constant and finetuning a few epochs** should let
the network re-equilibrate around a larger gate. The boost = 2 raw IoU
gain is consistent with this — γ_int boost helps at inference for the
raw path, the issue is only that the network can't tolerate
inference-time recalibration.

### Minimal-viable experiment

1. Load `runs/training/generator_v16_alignment_mirror/best_contact.pt`.
2. Re-initialize each layer's `gamma_int` parameter to a positive
   constant (sweep candidates: 0.1, 0.5, 1.0).
3. Finetune Stage B with v16 config for **5–10 epochs** (not from
   scratch — adapt around the new γ_int).
4. Evaluate: v17-E.20 config (per_step=20, Gumbel OFF) on the matched
   80-clip eval.

Wallclock estimate per init candidate: 5–10 epochs × ~10 min/epoch
= 1–2 h training + ~80 min eval = **~3 h per candidate**. 3 candidates
total ~9 h.

### Decision tree

| outcome | implication | next |
|---|---|---|
| γ_init=0.5 finetune improves contact + correct-part | γ_int IS the bottleneck; trained network can absorb larger gate when allowed to adapt | scale up: try γ_init=1.0 / longer finetune; eventually retrain Stage B from scratch with γ_init=0.5 |
| γ_init helps raw `full` but per-step `full_guided` doesn't compound | z_int helps raw path but per-step is at ceiling; ship "raw with finetuned γ" without per-step | restage v17-E baseline as "raw + boost" |
| All inits worse than v16 baseline | γ_int boost = 2 raw IoU gain was spurious; γ_int is not the lever | pivot to OMOMO-style explicit contact_target as input (architecture change but uses existing predictor output) |
| γ_init=0.1 already plateaus | small positive init suffices; further tuning unnecessary | finalise γ_init=0.1 as new default for any future Stage B training |

### Implementation cost

Code changes (rough estimate):

- `src/piano/models/motion_generator.py`: add `gamma_init_value: float = 0.0`
  parameter to the IntXAttn/encoder layer constructors. Currently
  hardcoded zero.
- New CLI entry / config field for "load ckpt + re-init γ_int + finetune":
  could be an additive flag on `train_generator.py`, e.g. `--reinit-gamma-int FLOAT`.
- New finetune config inheriting v16's: `configs/training/generator_v18_gamma_init_sweep.yaml`
  (or per-init).
- New sweep runner: `scripts/stage_b_generator/run_v18_gamma_init_sweep.sh`.

Total ~1 day implementation + 1 day server runs.

## References

- v17 series result docs:
  - `analyses/2026-05-01_per_step_guidance_design.md` (v17-C/D/E design baseline)
  - `analyses/2026-05-01_v17_per_step_result.md` (v17-C + v17-D/E sweep result)
  - `analyses/2026-05-01_v17_diagnostics_and_gumbel.md` (γ_int audit + Gumbel implementation)
  - `analyses/2026-05-01_v17f_gumbel_result_and_p1_plan.md` (v17-F negative + P1 plan)
- Run artefacts: `runs/eval/stageB_v0_17_v16bc_g_b{1,2,5,10,20}_*`.
- γ_int trajectories: `runs/wandb_logs/wandb_history_genB_v{14,15,16}*.csv`.
