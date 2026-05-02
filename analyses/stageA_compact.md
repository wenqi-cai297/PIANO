# Stage A Predictor — Compact History

Consolidates Stage A predictor evolution through 2026-05-03. Replaces
~10 dated docs (kept in git history for forensics if needed). Focus
is on **decisions + lessons**, not blow-by-blow.

## Predictor's role

Outputs 4 components consumed by Stage B as `z_int` conditioning via
[interaction_tokenizer.py:201](../src/piano/models/interaction_tokenizer.py#L201):

| field | shape | semantic |
|---|---|---|
| `contact_state` | (T, 5) sigmoid | which body parts contact when |
| `contact_target_attn` | (T, 5, 128) per-token sigmoid (v8.1+) | where on the object |
| `phase` | (T, 3) softmax | non_contact / stable_contact / manipulation |
| `support` | (T, 3) softmax (3-way after v9.1) | both_feet / single_foot / sitting |

True downstream signal of predictor quality: **γ_int convergence in
Stage B** (v4-v16 stuck at 0.02 = Stage B ignores z_int because it's
unreliable).

## Version trajectory (verdict per change)

| version | change | verdict | retained |
|---|---|---|---|
| v6 / v7 / v7-fix | architecture defaults; xyz regression on 5×3 | 21 cm L2 floor — **architectural normal**, not regression | ckpt = baseline reference |
| v8 (RETRACTED design) | KL on Gaussian softmax + DAG conditioning + Bengio TF + hinge consistency | NEGATIVE: phase regressed; topk_iou plateau; consistency loss ignored by optimizer | killed |
| v8.1 | MoMask random mask conditioning + multi-hot binary GT + focal+dice | PARTIAL WIN: phase F1 fixed (0.577→0.637); target <5cm hit 4.5%→11.6%; pelvis pct<10cm 33.9%→59.1%; foot regress | retained as base |
| v8.1.1 | top-K min mask + threshold-free top-K eval | small lift; foot still bad | retained |
| v9 | contact pos_weight + Mask3D 4-layer decoder + logit_adjust τ=1.0 | MIXED: foot recall 0→0.79 DOMINANT WIN via pos_weight; decoder no lift; τ=1.0 collapsed both_feet F1 0.94→0.000 | retained pos_weight; kept decoder as control |
| v9.1 | drop hand_support class + τ=1.0→0.3 | SHIP-READY: support 0.218→0.645, both_feet 0→0.91, hand recall +5pp; **but topk_iou still 0.13** | current production ckpt |
| **v9.2 (pending retrain)** | ASL contact + motion-aware trunk with random masking | targets contact precision (foot 0.06→0.20+) and topk_iou (0.13→0.25+) | this round |

## Key lessons (durable, not version-specific)

### L1. v6 baseline was 21 cm L2 — architectural normal
2026-05-04's "v7 21 cm disaster" claim was based on a fabricated v6
baseline (~5-10 cm). **v6 is also 21.13 cm** on the same metric.
21 cm L2 is what world-coordinate xyz regression on a 5 m manifold
gives with a small MLP head. It's not a bug; it's the architecture's
floor. **Always read the actual eval JSON before claiming regression.**

### L2. Bengio scheduled sampling is non-consistent
Huszár arXiv:1511.05101 (2015) proved scheduled sampling is not a
consistent estimator. Train-test gap is structural. v8 phase F1
0.577 was caused by TF distribution shift on the `contact_emb`
input to phase head. **Fix: random masking (MoMask CVPR 2024) — every
batch samples mask_ratio ~ Uniform[0,1], every information mix gets
trained.**

### L3. KL on Gaussian-softmax GT is not standard for HOI affordance
EgoChoir NeurIPS 2024, Text2HOI CVPR 2024 use **multi-hot binary GT
+ focal+dice on per-token sigmoid**. Multiple adjacent tokens are
valid contact targets — palm covers a region. Forcing softmax onto
single-token argmax over-constrains. **Fix: tokens within τ_part of
GT closest_xyz are positive; loss = focal + dice on the multi-hot mask.**

### L4. τ_part below FPS spacing → empty multi-hot mask
128 FPS tokens on a 1m object → ~8.8cm spacing. τ_foot=3cm produces
empty masks for most foot cells → vacuous loss → foot head untrained.
**Fix: GT mask = (top-K nearest tokens) ∪ (within-τ tokens), K=3.**

### L5. The "passive zero" pathology — pos_weight is sledgehammer
Contact head used bare BCE without pos_weight or focal across v6-v8.1.1
(focal_gamma in config only applied to phase / support CE). Foot
positive rate 3% → BCE dominated 32:1 by negatives → trivial "predict
zero" → recall = 0. Adding pos_weight cap=15 fixed recall but trashed
precision (foot precision 0.06, 17.5× over-predict). **Fix: ASL
(γ_pos=0, γ_neg=4, prob_shift=0.05) — preserves recall on positives
while focusing gradient on hard FPs.**

### L6. Logit Adjustment τ=1.0 over-corrects on extreme imbalance
Menon ICLR'21 says τ ∈ [0.5, 1.5]; we used τ=1.0 with 84.7 vs 3.1%
imbalance (27:1 ratio). Result: support both_feet F1 collapsed
0.94→0.000. **Fix: τ=0.3 for our extreme imbalance regime.**

### L7. Compound classes should be decomposed at extraction, not learned
hand_support = (hand_contact ∧ pelvis_static ∧ phase_stable). 3 % of
frames + 3-AND structure. Model never learned it (F1 ≈ 0 across all
versions). InterAct has no feet-airborne poses → hand_support is
"both_feet + hand bracing" — Stage B can derive it from
contact_state[hand]. **Fix: drop the class entirely (3-way support).**

### L8. Encoder upgrade is a trap on dense affordance
Heuken arXiv:2504.18355 ablates PointNet++ vs PT V3 on 3D AffordanceNet
— PT V3 *loses* 8.7 mIoU. PointNeXt and Sonata are similarly worse
on dense affordance vs scene-scale or whole-object tasks. **Don't
swap PointNet++.**

### L9. Token-level ranking has architectural ceiling at topk3_iou ~0.13
v8.1 (single-layer Q/K) → v8.1.1 (top-K mask) → v9 (Mask3D 4-layer
decoder, +7.7M params) → v9.1: topk3_iou stays at 0.12-0.14. Adding
head capacity didn't help. The bottleneck is **frame query lacks
per-frame body kinematics** — pelvis (stationary) works, hand/foot
(moving) fail. **Fix: motion-aware trunk (per-frame joint xyz input
to time tokens) — addresses W2 directly.**

### L10. Train-test feature asymmetry → MoMask random masking
Training has per-frame joints; inference doesn't (Stage B hasn't
generated motion yet). Bengio scheduled sampling is non-consistent
(L2). MoMask CVPR 2024 random masking (Uniform[0,1] mask ratio per
batch) trains a single model on every information mix, including
mask_ratio=1 which matches inference distribution → no shift.

## Current ship-ready ckpt (v9.1)

```
runs/training/predictor_v9_1_3way_support/best_val.pt  (ep 19)
```

Key absolute metrics (v9.1 best, val 1304 clips):

| component | metric | value | judgement |
|---|---|---:|---|
| contact (any_part) | F1 | 0.700 | recall 0.89, precision 0.58 |
| contact (foot) | precision | **0.06** | over-predicts 17×; v9.2 targets this |
| target (mask) | topk3_iou | 0.133 | 11× random; v9.2 targets this |
| target (centroid) | <5cm hit | 10.9% | adequate for "rough region" |
| phase | macro F1 | 0.620 | adequate (3-class) |
| support (3-way) | macro F1 | 0.645 | both_feet 0.91, sitting 0.67 |

By scenario:
- ✅ sit on chair (pelvis-dominant): production-grade
- ✅ single-/bi-manual grasp: usable, with hand_precision~0.4
- ⚠️ foot interactions: high recall but precision is noise
- ❌ fine-grained spatial precision: token-level ceiling at 0.13

## v9.2 pending fix

[analyses/2026-05-03_v92_asl_motion_aware_design.md](2026-05-03_v92_asl_motion_aware_design.md)
adds two orthogonal targeted fixes verbatim from upstream:

1. **ASL contact loss** ([Alibaba-MIIL/ASL](https://github.com/Alibaba-MIIL/ASL),
   797★) — fixes contact precision pathology
2. **Motion-aware trunk + MoMask masking**
   ([momask-codes](https://github.com/EricGuo5513/momask-codes), 1.3k★)
   — addresses topk_iou ceiling

32/32 sanity tests pass.

## v9.2 acceptance gates (vs v9.1)

| metric | v9.1 | v9.2 gate |
|---|---:|---|
| foot precision | 0.06 | ≥ 0.20 (FIX) |
| contact macro F1 | 0.37 | ≥ 0.50 |
| topk3_mean_iou | 0.13 | ≥ 0.25 |
| foot L2 | 40 cm | ≤ 30 cm (FIX) |
| contact any_part recall | 0.89 | ≥ 0.80 (preserve) |
| phase / support macro F1 | 0.62 / 0.65 | ≥ 0.60 / 0.60 (preserve) |

If passing → v8.1b Stage B refactor (consume `contact_target_attn`
directly) → v18 generator retrain.

If failing → v9.3 backlog candidates depending on which gate fails:
- foot precision still < 0.15 → ASL γ_neg=5 or per-part tuning
- topk3_iou still < 0.20 → cosine masking schedule + EgoChoir motion-KV
- foot recall regressed > 5pp → γ_pos > 0 needed for push-back

## Reference docs (NOT consolidated)

These remain as standalone references — they're surveys / specs, not
trial-and-error logs:

- [2026-05-02_alternatives_to_scheduled_sampling.md](2026-05-02_alternatives_to_scheduled_sampling.md) — TF alternatives survey
- [2026-05-02_class_imbalance_sota_survey.md](2026-05-02_class_imbalance_sota_survey.md) — long-tail / FP survey
- [2026-05-02_hoi_affordance_sota_survey_post_move_as_you_say.md](2026-05-02_hoi_affordance_sota_survey_post_move_as_you_say.md)
- [2026-05-02_hoi_data_aug_synthetic_transfer_survey.md](2026-05-02_hoi_data_aug_synthetic_transfer_survey.md)
- [2026-05-02_mtl_dag_research_survey.md](2026-05-02_mtl_dag_research_survey.md)
- [2026-05-02_predictor_v9_architecture_research.md](2026-05-02_predictor_v9_architecture_research.md)
- [2026-05-03_v92_asl_motion_aware_design.md](2026-05-03_v92_asl_motion_aware_design.md) — current pending v9.2 design
- [2026-05-03_pseudo_label_v12_strict_design.md](2026-05-03_pseudo_label_v12_strict_design.md) — v12 strict label spec
- [stageA_design.md](stageA_design.md) — Stage A v6 shipped state (legacy)
- [pseudo_label_pipeline.md](pseudo_label_pipeline.md) — label fields spec
