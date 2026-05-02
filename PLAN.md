# PIANO Plan

Compact action plan as of 2026-05-03.

## Immediate Priority — Stage A v9.2 server retrain

**Active branch**: 3-stage Stage A predictor refinement under v12 strict
labels. v9.1 ship-ready (3-way support, contact pos_weight, Mask3D
decoder, MoMask mask conditioning). v9.2 = ASL contact + motion-aware
trunk on top of v9.1, addressing the 2 specific remaining failures
(foot precision 0.06, topk3_iou 0.13). Detail:
`analyses/2026-05-03_v92_asl_motion_aware_design.md`.

### Execution sequence

1. **🟢 NEXT — v9.2 server retrain (~7 h, A6000 ×2)**
   ```bash
   cd /media/gpu-server-1/4TB_for_data/Cai/PIANO/PIANO
   git pull
   accelerate launch --config_file configs/accelerate_config.yaml \
     -m piano.training.train_predictor \
     --config configs/training/predictor_v9_2_asl_motion.yaml
   ```
   Startup log must show:
   ```
   [predictor cfg] structured_head.enabled = True
   contact pos_weight: ... (cap=...)        # NOT printed (use_contact_pos_weight=false in v9.2)
   contact_loss_kind: asl                    # ← if missing, ASL not active
   motion_aware_trunk.enabled = True         # ← if missing, plain trunk
   ```
2. **🟢 Eval + sync (3 files)**
   ```bash
   mkdir -p runs/eval/stageA_predictor_v9_2_asl_motion_val

   python scripts/stage_a_predictor/eval_predictor.py \
     --config configs/training/predictor_v9_2_asl_motion.yaml \
     --checkpoint runs/training/predictor_v9_2_asl_motion/best_val.pt \
     --split val \
     --output runs/eval/stageA_predictor_v9_2_asl_motion_val/predictor_v9_2_asl_motion_val_best.json

   python scripts/stage_a_predictor/eval_predictor.py \
     --config configs/training/predictor_v9_2_asl_motion.yaml \
     --checkpoint runs/training/predictor_v9_2_asl_motion/final.pt \
     --split val \
     --output runs/eval/stageA_predictor_v9_2_asl_motion_val/predictor_v9_2_asl_motion_val_final.json

   python scripts/stage_a_predictor/dump_wandb_history.py \
     --name predictor_stageA_v9_2_asl_motion \
     --output runs/wandb_logs/wandb_history_predictor_v9_2_asl_motion.csv
   ```

### v9.2 acceptance gates (vs v9.1)

| metric | v9.1 | v9.2 gate |
|---|---:|---|
| foot precision | 0.06 | ≥ 0.20 (FIX) |
| contact macro F1 | 0.37 | ≥ 0.50 |
| topk3_mean_iou | 0.13 | ≥ 0.25 |
| foot L2 (cm) | 40 | ≤ 30 (FIX) |
| contact any_part recall | 0.89 | ≥ 0.80 (preserve) |
| phase / support macro F1 | 0.62 / 0.65 | ≥ 0.60 / 0.60 (preserve) |

Pass = 4/4 critical + 3/4 preserve.

## After v9.2 acceptance — Stage B v8.1b refactor (~2-3 days)

Refactor `src/piano/models/interaction_tokenizer.py` to consume
`contact_target_attn (B, T, 5, 128)` per-token sigmoid mask directly
instead of `contact_target_xyz (B, T, 5, 3)` flat concat. Two design
options (TBD per `analyses/stageA_compact.md` §v9.2 backlog):

- (a) Flat concat: 640 dim → z_int. Simple but bloats tokenizer input.
- (b) Parallel obj-affordance cross-attn alongside IntXAttn (EgoChoir
      motion-stream KV style). More principled but larger refactor.

Touches: `interaction_tokenizer.py`, `motion_generator.py` IntXAttn,
`contact_guidance.py` per-step, `inference/generate.py`.

Then: train v18 generator on v9.2 predictor outputs, eval with
unified metric set (penetration, weighted_local, correct_part_recall,
soft_IoU, jerk + KS).

## v9.2 fallback paths (if acceptance fails)

| failure | candidate fix |
|---|---|
| foot precision still < 0.15 | ASL γ_neg=5 or per-part γ_neg tuning |
| topk3_iou still < 0.20 | cosine masking schedule (instead of uniform) + EgoChoir motion-KV stream as parallel KV |
| recall regressed > 5pp | γ_pos > 0 or modest pos_weight floor combined with ASL |

## Long-term backlog

- **Stage C joint finetune** — never started; on roadmap. Joint train
  predictor + generator + extractor via consistency loss. Best after
  Stage B v18 evidence shows γ_int growing.
- **EgoChoir motion-KV parallel stream** — true 2-stream architecture
  rather than time-token augmentation. v9.x candidate if v9.2 still
  doesn't lift topk_iou enough.
- **Object encoder pretraining** — Heuken arXiv:2504.18355 ablation
  says NOT to upgrade to PT V3 / Sonata for dense affordance, but
  pretraining PointNet++ on ShapeNet might help.

## Predicted v18 outcome (post v9.2 + v8.1b)

- raw correct_part_recall: 0.176 (v16) → 0.30+
- guided correct_part_recall: 0.292 (v17-E.50) → 0.40+
- γ_int convergence value: 0.02 (v4-v16) → 0.08+ (z_int now informative)
- Visual: real contact instead of "approach"

If γ_int < 0.05 after v18, predictor still inadequate → escalate to
v9.x or v10 architecture.
