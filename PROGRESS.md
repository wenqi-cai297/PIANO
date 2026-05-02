# PIANO Progress

Compact state as of 2026-05-03. Detailed history is in
`analyses/stageA_compact.md` and `analyses/stageB_compact.md`.

## §0 Active long runs

(Nothing currently running on server. v9.2 ready to launch — see PLAN.md.)

## §1 Stage state at a glance

| stage | state | ckpt of record |
|---|---|---|
| Stage 1 (pseudo-labels) | ✅ v12 strict shipped | `<subset>/pseudo_labels/v12_strict/` |
| Stage A (predictor) | ✅ v9.1 ship-ready · 🟡 v9.2 prototype landed, awaiting server retrain | `runs/training/predictor_v9_1_3way_support/best_val.pt` |
| Stage B (generator) | ⏸️ blocked on Stage A v9.2 + v8.1b refactor | `runs/training/generator_v14_sampled_st_contact/best_contact.pt` (last saved) |
| Stage C (joint) | ⏸️ never started | — |

## §2 Key Stage A absolute metrics (v9.1 best, val 1304 clips)

| component | metric | value | judgement |
|---|---|---:|---|
| contact (any_part) | F1 | 0.700 | recall 0.89, precision 0.58 |
| contact (foot) | precision | 0.06 | over-predicts 17×; v9.2 ASL targets this |
| target (mask) | topk3_mean_iou | 0.133 | 11× random; v9.2 motion-aware targets this |
| target (centroid) | <5cm hit | 10.9% | "rough region" only |
| phase | macro F1 | 0.620 | adequate |
| support (3-way) | macro F1 | 0.645 | both_feet 0.91, sitting 0.67 |

Verdict: ship-ready for "sit + grasp" scenarios; foot precision and
fine spatial token ranking are v9.2 targets.

## §3 Stage B status

Last evaluated checkpoint:
`runs/training/generator_v14_sampled_st_contact/best_contact.pt` —
matched 80-clip eval contact 27.37 cm (raw), v14 K=16 composite
oracle 17.94 cm + moving_coupled 0.3715. Distinct from v17-E.50 which
was the local 1-sample SOTA via per-step guidance (16.50 cm) but had
metric-gaming flags (penetration, jerk×8). See
`analyses/stageB_compact.md` for the full v0-v17 trajectory.

v18 (next Stage B retrain) is queued behind Stage A v9.2 acceptance +
v8.1b InteractionTokenizer refactor (consume `contact_target_attn`
mask directly instead of xyz, per Path B selected 2026-05-03).

## §4 Recent decisions (durable, not blow-by-blow)

- **2026-05-03 — v12_strict labels accepted.** v11 "approach within
  12cm" replaced by v12 strict "real contact" definition (5cm palm +
  duration + engagement, per-part τ). PC-eval frame frac 78% → 45%.
  Detail: `analyses/2026-05-03_pseudo_label_v12_strict_design.md`.
- **2026-05-03 — ship metric pivot.** Unified metric set (penetration,
  weighted_local, soft IoU, jerk + KS) introduced after v17-E.50 was
  flagged for metric gaming. Default ship recipe is now v17-E.20 +
  final.pt (less aggressive per-step). Detail:
  `analyses/2026-05-03_unified_metric_results.md`.
- **2026-05-03 — Stage A absolute audit.** v9.1 predictor adequate for
  shipping but with known weaknesses (foot precision 0.06, topk_iou
  0.13, no fine spatial ranking). Decision: pre-fix via v9.2
  (ASL contact + motion-aware trunk) BEFORE Stage B v18 retrain.
  Detail: `analyses/stageA_compact.md`.

## §5 Next work (current branch)

See PLAN.md `## Immediate Priority` for executable commands.

Sequence:

1. ✅ v9.2 prototype committed (32/32 tests pass).
2. 🟢 v9.2 server retrain (~7 h).
3. 🟢 v9.2 eval against acceptance gates.
4. 🟢 If pass → v8.1b Stage B InteractionTokenizer refactor (~2-3 days)
   → v18 generator retrain (~1 day) → unified-metric eval.
5. 🟢 If v18 visual passes → ship v18 + per-step inference as default.
6. ⏸️ Stage C joint finetune as long-term backlog.
