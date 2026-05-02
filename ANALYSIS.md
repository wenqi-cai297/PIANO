# PIANO Analysis Index

Compact index for durable analysis docs. Old dated docs (v17 era,
v6/v7/v8 trial-and-error) were consolidated into compact summaries
on 2026-05-03 to reduce context load.

## Compact summaries (read these first)

| File | Purpose |
|---|---|
| `analyses/stageA_compact.md` | Stage A predictor v6 → v9.2 evolution: per-version verdict + 10 durable lessons. Read before touching predictor. |
| `analyses/stageB_compact.md` | Stage B generator v0 → v17 evolution: training/inference/eval branches + decisions. |
| `analyses/early_setup.md` | Server / data / backbone setup gotchas. |
| `analyses/pseudo_label_pipeline.md` | Stage 1 label fields, thresholds, abandoned paths. |
| `analyses/stageA_design.md` | Stage A v6 shipped state (legacy reference). |

## Specs / current designs

| File | Purpose |
|---|---|
| `analyses/2026-05-03_pseudo_label_v12_strict_design.md` | v12 strict label definition (current production). |
| `analyses/2026-05-03_v92_asl_motion_aware_design.md` | **Stage A v9.2 — current pending retrain.** ASL contact loss + motion-aware trunk with MoMask random masking. |
| `analyses/2026-05-01_per_step_guidance_design.md` | v17 per-step decoded-geometric guidance design (Stage B inference). |
| `analyses/2026-05-02_codec_floor_baselines.md` | VQ codec floor on alignment metrics — paradigm shift. |
| `analyses/2026-05-03_unified_metric_results.md` | Unified Stage B ship metrics (penetration / weighted_local / soft IoU / jerk). |
| `analyses/2026-05-03_gamma_int_re_evaluation.md` | γ_int trajectory v4 → v16 + downstream interpretation. |

## Frontier surveys (read on demand for new design decisions)

| File | When to consult |
|---|---|
| `analyses/2026-05-02_alternatives_to_scheduled_sampling.md` | Train-test gap, masking, scheduled sampling alternatives. |
| `analyses/2026-05-02_class_imbalance_sota_survey.md` | Long-tail / FP / ASL / pos_weight tradeoffs. |
| `analyses/2026-05-02_hoi_affordance_sota_survey_post_move_as_you_say.md` | HOI affordance prediction architectures (EgoChoir, Text2HOI, etc.). |
| `analyses/2026-05-02_hoi_data_aug_synthetic_transfer_survey.md` | Data augmentation, synthetic data, transfer learning. |
| `analyses/2026-05-02_mtl_dag_research_survey.md` | Multi-task DAG / gradient conflict / consistency loss. |
| `analyses/2026-05-02_predictor_v9_architecture_research.md` | Architecture options (encoders, decoders, heads). |

## Root memory docs

| File | Purpose |
|---|---|
| `restart_prompt.md` | Fast recovery checklist for fresh sessions. |
| `PROGRESS.md` | Current numbers + active branches + recent decisions. |
| `PLAN.md` | Next executable actions. |
| `SPEC.md` | Stable project design + code layout. |
| `SUGGESTION.md` | Current recommendation memo. |

## Hygiene

When a new experiment finishes:

1. Update `PROGRESS.md` (snapshot section) and `PLAN.md` (next action).
2. If the result changes a Stage's direction, append to or revise the
   stage's compact doc (`stageA_compact.md` / `stageB_compact.md`).
3. Avoid creating new dated `analyses/YYYY-MM-DD_*.md` unless the doc
   is a multi-page design / survey that won't fit in the compact doc.
4. Do not create per-version dated docs (`v9_design`, `v9.1_results`,
   etc.) — they accumulate and create noise. Update the compact doc
   instead.
5. Once a result is stable, cite via the compact doc, not the dated
   intermediate.
