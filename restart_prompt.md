# Restart Orientation

Quick recovery checklist for fresh sessions. **Don't trust this file
alone — verify against `git log -10 --oneline` + the current state of
`PROGRESS.md` / `PLAN.md`.**

## First commands

From repo root:

```bash
git status --short --branch
git log -10 --oneline
ls runs/training/ 2>/dev/null | head -10
```

## Read order (priority)

1. **`PROGRESS.md`** — current snapshot, key absolute metrics, recent decisions
2. **`PLAN.md`** — next executable actions
3. **`analyses/stageA_compact.md`** — Stage A predictor history + lessons
4. **`analyses/stageB_compact.md`** — Stage B v0-v17 trajectory
5. **`ANALYSIS.md`** — index into reference docs (surveys, design docs)
6. **`SPEC.md`** — stable project design (read once)

Reference docs (read on demand):
- `analyses/2026-05-03_v92_asl_motion_aware_design.md` — current pending Stage A retrain
- `analyses/2026-05-03_pseudo_label_v12_strict_design.md` — v12 label spec
- `analyses/2026-05-02_*_survey.md` — 6 frontier-paper surveys (TF alternatives, class imbalance, HOI affordance, MTL DAG, predictor architecture, data aug)

## Current state, 2026-05-03

**Active branch**: Stage A v9.2 — ASL contact loss + motion-aware
trunk with MoMask random masking. v9.1 is ship-ready (production
ckpt below); v9.2 targets the 2 specific remaining failures.

**Production Stage A predictor** (current `z_int` source):
```
runs/training/predictor_v9_1_3way_support/best_val.pt  (ep 19)
```
Key metrics:
- contact any_part F1 0.70 (recall 0.89, precision 0.58)
- target topk3_iou 0.133, centroid <5cm hit 10.9%
- phase macro F1 0.62, support 3-way macro F1 0.645
- foot precision 0.06 ← v9.2 ASL targets this
- topk3_iou plateaus 0.13 ← v9.2 motion-aware trunk targets this

**Stage B**: blocked. Last evaluated ckpt
`runs/training/generator_v14_sampled_st_contact/best_contact.pt`.
v18 retrain queued behind Stage A v9.2 acceptance + v8.1b refactor.

**Next action** (PLAN.md §Immediate Priority):
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
  -m piano.training.train_predictor \
  --config configs/training/predictor_v9_2_asl_motion.yaml
```

## Hard-won lessons (durable, applicable to any future iteration)

These are in `analyses/stageA_compact.md` "Key lessons" but worth
front-loading here:

1. **Always read the actual eval JSON before claiming regression.** The
   "v7 21cm L2 disaster" was based on a fabricated v6 baseline; v6 was
   also 21cm. Cross-check before fixing imaginary problems.
2. **Bengio scheduled sampling is non-consistent (Huszár 2015).** Use
   MoMask random masking (Uniform[0,1] mask ratio per batch) for
   train-test asymmetry — every information mix gets trained.
3. **Multi-hot binary GT + focal+dice is HOI literature standard**, not
   KL on Gaussian softmax. EgoChoir / Text2HOI / DECO all use this.
4. **pos_weight is a sledgehammer**: helps recall, trashes precision.
   For multi-label binary with extreme imbalance, ASL (γ_pos=0,
   γ_neg=4, prob_shift=0.05) is the proper fix.
5. **Logit Adjustment τ=1.0 over-corrects on extreme imbalance.** Use
   τ ∈ [0.3, 0.5] for our 27:1 ratio dataset.
6. **Compound classes**: don't try to learn `A ∧ B ∧ C` end-to-end at
   3% positive rate. Decompose at extraction (e.g., we dropped
   hand_support entirely; Stage B can derive it from contact_state[hand]).
7. **Encoder upgrade is a trap on dense affordance.** Heuken
   arXiv:2504.18355 — PointNet++ wins by 8.7 mIoU over PT V3.
8. **Token-level ranking ceiling at ~0.13 is architectural.** Adding
   head capacity (Mask3D 4-layer +7.7M params) didn't help. The fix
   is making per-frame body kinematics directly available to the trunk
   (motion-aware trunk in v9.2).
9. **Sanity tests catch DDP / unused-parameter bugs early.** Any new
   trainable parameter under DDP must receive gradient at backward —
   we caught 2 such bugs (MHA out_proj v8 → v8.1, part_queries
   v9 → v9.1) only because tests check this explicitly.
10. **One change per commit + retrain.** Multi-knob changes break
    attribution. v9 combined ASL + Mask3D + logit_adjust into one
    retrain — couldn't tell which helped or hurt until a 2nd retrain
    isolated each.

## Hardware

2 × A6000 48 GB. bf16 mixed precision via HuggingFace Accelerate DDP.
Predictor train ~6-7 h; Stage B generator train ~24 h.

## Common gotchas

- **`git status` shows `analyses/` untracked**: yes, gitignored. Use
  `git add -f analyses/<file>.md` to commit. Already-committed files
  in `analyses/` show normally.
- **Accelerate launch on Windows**: dev environment is Windows
  PowerShell, training environment is Linux server. Don't try to
  launch on Windows.
- **`piano-train-predictor` vs `python scripts/...`**: the former is a
  console script (`accelerate launch -m piano.training.train_predictor`
  is the canonical form for training); scripts/stage_a_predictor/*.py
  are direct-invocation utilities.
