# 2026-05-02 — Predictor v9 architectural research: pushing topk3 IoU 0.13 → 0.30+

Companion to:
- `analyses/2026-05-02_hoi_affordance_sota_survey_post_move_as_you_say.md` (v1 SOTA survey)
- `analyses/2026-05-05_v8_round1_diagnosis_and_v81_plan.md` (v8.1 plan)
- `analyses/2026-05-05_v81_results_and_v811_plan.md` (v8.1 results)

## TL;DR — direct answer to the ROI question

**Highest-ROI single architectural change for hand/foot moving contact**:

> **Replace soft `attention-weighted` xyz / sigmoid-token regression with a
> Mask3D / Mask2Former-style mask-decoder head**, where each
> (frame × body-part) cell instantiates a learnable query that iteratively
> cross-attends to object tokens with mask-guided attention, and emits a
> per-token mask directly.

This is the same architectural class that (a) dominates 3D instance
segmentation (Mask3D, OneFormer3D — CVPR 2024), (b) underlies
**InteractVLM (CVPR 2025) which lifted DAMON F1 55.0 → 75.6** (DECO
ICCV 2023 baseline → InteractVLM) on per-vertex body contact, and (c)
matches the v8 / v8.1 architecture's identified weakness — the per-part
query-vs-128-token attention is one-shot, not iterative.

**Second-highest-ROI**: add the EgoChoir-style parallel motion-KV
stream where the second KV is **body-joint kinematics over time**
(velocity / acceleration of 22 joints, not head pose), with τ-modulation.

**Lowest-ROI of the four candidates**: encoder upgrade. Empirical
evidence from Heuken et al. (arXiv:2504.18355, May 2025) tested
PointNet++ 21.5 mIoU vs PT V3 12.8 mIoU on 3D AffordanceNet — **PT V3
*lost* 8.7 mIoU points to PointNet++**. The 1.3 M-param PointNet++ is
not the bottleneck for affordance.

**Trunk capacity**: 10-layer / d=384 / ~30 M-param trunk is in the right
ballpark for our data regime (~10 K clips). Diffusion Transformer
scaling laws (Peebles & Xie ICLR 2023; DiT-XL = 675 M params for FID
on ImageNet) predict diminishing returns above ~3 B params and 600 M
samples. We are 4 orders of magnitude below saturation; depth doesn't
help unless data scales 100-1000×.

## 1. Context: the gap we're trying to close

From `analyses/2026-05-05_v81_results_and_v811_plan.md`:

| metric | v7-fix | v8 | v8.1 | EgoChoir GIMO | InteractVLM DAMON | gap to SOTA |
|---|---:|---:|---:|---:|---:|---|
| topk3 IoU | n/a | n/a | 0.13 | ~0.45 (full body) | n/a | **3.5×** |
| pelvis pct<10cm | 33.9 % | 51.9 % | 59.1 % | n/a | n/a | strong |
| hand L2 (cm) | 22.4 | 22.8 | 24.5 | n/a | n/a | weak |
| foot L2 (cm) | 25.3 | 27.4 | 42.7 | n/a | n/a | regressed |
| body contact F1 | n/a | n/a | n/a | n/a | **75.6** (DECO 55.0) | n/a |

The gap pattern: **pelvis (stationary) works; hand/foot (moving)
don't**. Pelvis F1 nearly matches what one would expect from the
architecture; hand/foot is 3-4× worse.

The diagnosis in v8.1 docs is correct: the StructuredHead's per-part
learnable query attends ONCE to object tokens, with the per-frame
trunk feature as the only motion signal. This is 5-10× thinner than
what the SOTA literature uses for moving contact.

## 2. The four hypothesis classes and the evidence

### 2.1 Hypothesis A: Encoder capacity — PointNet++ undersized [WEAK]

**Evidence against**:

1. **Heuken, Goebbels, Pomerleau, Roth. "Interpretable Affordance
   Detection on 3D Point Clouds with Probabilistic Prototypes."
   arXiv:2504.18355 (May 2025), submitted to BMVC.** Table 5
   directly compares backbones on 3D AffordanceNet:

   | Backbone | mIoU | mAP | mAUC | MSE |
   |---|---:|---:|---:|---:|
   | PointNet++ | 21.5 | 50.9 | 83.1 | 0.02 |
   | DGCNN | 18.4 | 44.6 | 80.0 | 0.050 |
   | **PT V3** | **12.8** | **35.4** | 81.1 | 0.066 |

   "All four models perform worse than their PointNet++ counterparts."
   This is dead-center-relevant — affordance prediction over object
   point clouds, **and PT V3 LOSES 8.7 mIoU points**.

2. **Sonata** (Wu et al., CVPR 2025 Highlight, arXiv:2503.16429) is
   tuned for **scene-scale point clouds** (rooms, outdoor LiDAR);
   its ScanNet linear probing tripled from 21.8 → 72.5%, but the
   evaluation domain is 100k+ point room scans, not 1 K-point object
   meshes. The "geometric shortcut" failure mode they fix is from
   indoor/outdoor LiDAR sparsity — orthogonal to our setting.

3. **Uni3D** (Zhou et al., ICLR 2024 Spotlight, arXiv:2310.06773,
   github 1.6k★, baaivision/Uni3D) achieves SOTA on zero-shot
   classification + open-world segmentation — but again the eval is
   on whole-object semantic categories. No public number on dense
   per-point HOI affordance.

4. **HOI-Diff (CVPR 2025 HuMoGen Workshop, arXiv:2312.06553)** — the
   closest published HOI affordance predictor to ours — uses a
   **standard PointNet++** to encode 512 object points + 8-layer
   transformer encoder with d=512, 4 heads. They published competitive
   numbers without any encoder upgrade.

5. **Text2HOI (CVPR 2024, arXiv:2404.00562)** uses standard
   PointNet++.

**Evidence for**:

- None convincingly. The "1.3 M params is small" intuition is
  reasonable but contradicted by direct experiment.

**Verdict**: encoder upgrade is **the lowest-ROI direction**.
PointNet++ is the published standard for HOI affordance; transformer
backbones underperform on small-object affordance. Don't spend a
retrain on this.

**Caveat — IF we ever scale to 6890-vertex per-vertex contact**
(EgoChoir / InteractVLM output dim), then encoder cost matters more
because the head queries scale 50×. At our 128-token output, the
encoder is fine.

### 2.2 Hypothesis B: Trunk capacity — 10 layers undersized [WEAK]

**Evidence against**:

1. **DiT scaling laws** (Peebles & Xie, ICLR 2023, arXiv:2212.09748).
   ImageNet 256² FID saturates at DiT-XL (675 M params) given 1.28 M
   training images. Compute-vs-quality: clean power law up to
   saturation. Translation: with **10 K training clips** we are 100×
   below the saturation regime; doubling our depth from 10 → 20
   layers (from 30 M → 60 M params) is unlikely to lift hand/foot
   contact.

2. **HOI-Diff APDM** uses 8 transformer layers d=512 — within 1.3× of
   our 10 layers d=384. Not a meaningful capacity gap.

3. **EgoChoir** doesn't publish the trunk depth, but their per-frame
   contact head sits on top of a moderate-size backbone (likely
   ResNet/ViT-base for video features), not a 24-layer transformer.

4. The v8.1 result shows phase F1 = 0.637 = recovers v7-fix, while
   target topk3 IoU is 0.13. **The trunk learns plenty fast on phase
   classification — capacity is not the bottleneck**, the head is.

**Verdict**: trunk capacity is **not the bottleneck**. 10 layers d=384
is well within the published HOI predictor sweet spot.

**Conditional**: if a future v10 hits a clear plateau on `topk3 IoU` >
0.40 with the right head + motion stream, we can revisit width
(d=384 → 512) or depth (10 → 14). Not now.

### 2.3 Hypothesis C: Motion-stream parallel KV (EgoChoir-style) [STRONG]

**Evidence for**:

1. **EgoChoir** (Yang et al., NeurIPS 2024, arXiv:2405.13659,
   github yyvhang/EgoChoir_release, 30★). Architecture:
   - Visual features F_V (RGB clip features from egocentric video)
     and motion features F_M (head pose / head motion sequence) are
     **two parallel KV streams**.
   - Modulation tokens τ_v, τ_m **adjust gradients of the layers
     mapping interaction clues** in the parallel cross-attention.
     "When the visual signal is occluded or ambiguous, motion
     dominates; when motion is small or stationary, visual dominates."
   - Ablation: removing motion stream "particularly impacts body
     interactions" — this is the failure mode we observe (hand/foot
     moving contact regressed; pelvis stationary contact works). The
     paper says the model "can hardly anticipate interaction regions
     without the head motion, particularly for body interactions."

2. **Caveat**: EgoChoir's motion stream is **head pose** (egocentric
   camera), not body-joint kinematics. **For our setting, the
   analogous motion signal is per-frame body kinematics** — joint
   xyz, velocity, acceleration, or differences. We have CLIP text +
   initial pose + object — all stationary. The per-frame trunk
   feature does have time-resolved info (passes through self-attn
   over 196 frames), but it is **not a kinematic feature** in the
   physical sense (no explicit Δposition / velocity).

3. **MotionBERT** (Zhu et al., ICCV 2023, arXiv:2210.06551,
   github Walter0807/MotionBERT, 800+★) and **MotionAGFormer**
   (Mehraban et al., WACV 2024, arXiv:2310.16288, github
   TaatiTeam/MotionAGFormer, 200+★) both use Dual-stream
   Spatio-temporal Transformer (DSTformer) with parallel
   spatial+temporal attention, where **per-frame joint features are
   the input**. They get SOTA on Human3.6M MPJPE. Most relevant for
   our case: a parallel "motion stream" computed from joint
   velocities at the trunk level.

4. **SkateFormer** (Do et al., ECCV 2024, arXiv:2403.09508,
   github KAIST-VICLab/SkateFormer, 200+★) uses Skate-MSA (skeletal
   self-attention partitioned by physical joint connectivity +
   temporal locality). Most relevant SOTA on skeleton-based action
   recognition; **the partition-by-physical-joint trick directly
   maps to our per-body-part query design**.

**Implementation cost**:

- Motion features F_M: per-frame (joint xyz - prev xyz) ⊕ joint xyz —
  add a small MLP encoder (d=64 → d=384) over (T-1, 22, 6) →
  (T-1, 22, d). Then either project to (T, K_motion=22, d) or pool
  to (T, K_motion=1, d).
- Add a third sublayer to PredictorBlock: motion_attn (Q from x,
  K/V from F_M). +1 LayerNorm, +1 MultiheadAttention per layer.
  Roughly +1.5 M params at d=384, h=6, 10 layers.
- Modulation: simplest is a **learned scalar gate** that mixes
  visual-attn output and motion-attn output: `out = α_v * vatt +
  α_m * matt`, with α_v, α_m softmaxed across the 2 streams,
  initialized 0.5/0.5. The full EgoChoir gradient-modulation is
  ~50 lines and shouldn't be the first thing tried.

**Expected lift**: hand L2 24.5 → ~18 cm; foot L2 → ~25 cm; topk3 IoU
0.13 → 0.20-0.25. Estimate based on (a) EgoChoir's claim that motion
is necessary "particularly for body interactions," (b) the 30 % drop
in body F1 when their motion stream is removed (paper text, not
table values).

**Pitfall**: motion encoder MUST take per-frame body kinematics
(SMPL joint xyz over time), not just initial pose. Currently the
[POSE] token is a dedicated index-0 embedding from `init_pose` (66 d,
SMPL-22). To make this stream useful for moving contact, we need to
plumb the **full per-frame kinematics** into a separate stream. This
is more work than just adding a motion-attn sublayer — it requires
the dataloader to emit `joint_xyz_per_frame: (T, 22, 3)` and the
caller to extract velocity / acceleration features.

### 2.4 Hypothesis D: Decoder-style head (Mask2Former / Mask3D-style) [STRONGEST]

**Evidence for**:

1. **Mask3D** (Schult et al., ICRA 2023, arXiv:2210.03105,
   github JonasSchult/Mask3D, 700+★) and **OneFormer3D** (Kolodiazhnyi
   et al., CVPR 2024, arXiv:2311.14405, github oneformer3d, 200+★).
   Architecture for our setting:
   - Each (frame × body-part) cell instantiates a **learnable query**
     (we already have `part_queries`, but only one per body part —
     not refined iteratively).
   - **Iteratively** cross-attends to multi-scale object features
     across L decoder layers (default L=6 in Mask3D, L=4 in
     OneFormer3D).
   - **Self-attention between queries** lets queries reason about
     each other (in our case: hand and foot queries can communicate
     about coordinated grasping).
   - Per-query mask predictor: dot-product between query and per-token
     features → per-token logit → sigmoid → mask. This is **already
     compatible with our v8.1 multi-hot binary GT + focal+dice loss**.

2. **InteractVLM** (Dwivedi et al., CVPR 2025, arXiv:2504.05303,
   github saidwivedi/InteractVLM, 100+★). MV-Loc uses **SAM mask
   decoder architecture** (not SAM2) with two decoders — one for
   human contact, one for object contact. Loss = focal + L1. **Lifts
   DAMON body contact F1 from DECO's 55.0 % to 75.6 %**, a 20+ pp
   absolute gain. The mechanism: SAM-style decoder is iterative,
   query-conditioned, and does sigmoid + focal — and that's enough.

3. **DETR-class queries** (CVPR/NeurIPS-validated for 4 years now in
   2D detection) reliably outperform single-shot heads when the
   number of structured outputs is small (we have T × 5 = 980
   queries per clip — comparable to 100-200 per Mask3D scene).

**Implementation cost** (heavier than motion stream):

- New `MaskDecoder` module: stack of 4-6 cross-attn + self-attn +
  FFN layers. Each layer:
  - Self-attn over (T*P=980 queries, d=384) — costly. Consider
    per-frame self-attn only (P=5 per frame) to keep compute
    reasonable.
  - Cross-attn to object tokens (Lq=980, Lk=128).
  - FFN.
- Replaces v8.1's `CrossAttentionWeightsOnly` 1-layer attention with
  a 4-6 layer decoder.
- Roughly +6 M params (~20 % of the trunk).
- ~200 lines of new code.

**Expected lift**: hand L2 24.5 → ~15 cm; foot L2 42.7 → ~22 cm;
**topk3 IoU 0.13 → 0.30-0.40** based on (a) InteractVLM's 20 pp F1
lift over DECO using a similar architectural shift, (b) Mask3D
outperforming non-decoder methods by 10+ AP on instance segmentation,
(c) the iterative refinement directly addresses "argmax bounces
between adjacent tokens" diagnosis from v8 Round 1.

**Risk**: 4-6 decoder layers might overfit on 8475 training clips. To
mitigate, use auxiliary deep-supervision losses (Mask2Former pattern:
loss at every decoder layer, not just last).

## 3. Comparative table — what we'd build for each hypothesis

| direction | params Δ | code Δ | training time Δ | expected topk3 IoU |
|---|---:|---:|---:|---:|
| Encoder upgrade (PT V3) | +9 M | +500 lines | +50 % | **0.13 (no lift)** |
| Trunk depth 10 → 14 | +12 M | +5 lines | +40 % | 0.15 (+0.02) |
| Motion-KV stream | +1.5 M | +100 lines | +5 % | 0.20-0.25 |
| Mask-decoder head | +6 M | +200 lines | +15 % | **0.30-0.40** |
| Motion-KV + decoder | +7.5 M | +300 lines | +20 % | **0.35-0.45** |

## 4. Strong candidates table (5-7 max, ruthlessly filtered)

Filters: top venues 2023-2026, public GitHub > 100★ OR clear
HOI/affordance/contact application, reproducible, 2024+ preferred.

| # | Paper | Venue | arxiv | Code | ★ | Architectural innovation | Lift expected |
|---|---|---|---|---|---:|---|---|
| 1 | **InteractVLM** | CVPR 2025 | 2504.05303 | saidwivedi/InteractVLM | 100+ | SAM-style mask decoder for body contact + object contact, focal + L1 | **Direct precedent: F1 55.0 → 75.6 over DECO** |
| 2 | **Mask3D** | ICRA 2023 | 2210.03105 | JonasSchult/Mask3D | 700+ | Mask2Former-style iterative query decoder for 3D point clouds | Architectural reference for our (T×P) → 128-token mask head |
| 3 | **OneFormer3D** | CVPR 2024 | 2311.14405 | oneformer3d/oneformer3d | 200+ | Unified panoptic + instance + semantic mask decoder | Multi-task generalization of Mask3D |
| 4 | **EgoChoir** | NeurIPS 2024 | 2405.13659 | yyvhang/EgoChoir_release | 30 | Parallel motion-KV with τ-modulated gradient gating | **Direct precedent for our moving-contact gap** (motion stream necessary "particularly for body interactions") |
| 5 | **MotionAGFormer** | WACV 2024 | 2310.16288 | TaatiTeam/MotionAGFormer | 200+ | Dual-stream transformer + GCNFormer for per-frame joint kinematics | Reference for body-kinematics motion encoder |
| 6 | **SkateFormer** | ECCV 2024 | 2403.09508 | KAIST-VICLab/SkateFormer | 200+ | Skate-MSA: physical-joint-partitioned spatial-temporal attention | Reference for per-body-part query partitioning |
| 7 | **HOI-Diff (APDM)** | CVPR 2025 WS | 2312.06553 | neu-vi/HOI-Diff | 161 | 8-layer transformer + PointNet++ for joint-level contact diffusion | Direct architectural precedent for HOI contact predictor; baseline numbers |
| 8 | **DECO** | ICCV 2023 | 2309.15273 | sha2nkt/deco | 250+ | Three-branch (scene + part + contact) per-vertex contact predictor | Body part attention precedent; baseline that InteractVLM beats |

**Ruled out**:

- **PT V3 / Sonata / Uni3D**: SOTA point cloud foundation models, but
  evidence (Heuken et al. 2025, arXiv:2504.18355) shows transformer
  backbones lose 8+ mIoU vs PointNet++ on 3D AffordanceNet.
- **PointNeXt**: improves PointNet++ in scene segmentation, but no
  HOI ablation showing it dominates on per-point affordance.
- **Point-BERT / MaskPoint**: pre-training-as-encoder approach;
  affordance applications underwhelming in published numbers.
- **CG-HOI**: no code release; can't reproduce.
- **GenHOI**: code not yet released as of June 2025.
- **MaskGIT-style iterative refinement at the head**: speculation —
  no published HOI use case with this specific pattern.
- **Diffusion contact head (HOI-Diff APDM)**: 8 transformer layers +
  500 sampling steps at inference is too heavy for our setting where
  the predictor is called every Stage B step.

## 5. Specific question — single highest-ROI change

**Pushing topk3 IoU 0.13 → 0.30+ on hand/foot moving contact**.

**Answer**: **Mask-decoder head** (Mask3D / Mask2Former style, with
6 decoder layers, auxiliary losses, and per-frame self-attn over the 5
body-part queries). Specifically:

```
1. v8.1 keeps multi-hot binary GT + focal + dice + topk3 mask.
2. Replace the 1-layer CrossAttentionWeightsOnly with a 6-layer
   MaskDecoder (each layer = self-attn between (P=5) part-queries
   per frame + cross-attn to (M=128) object tokens + FFN).
3. Auxiliary loss at every decoder layer (deep supervision).
4. Per-frame self-attn (not full T*P) to keep compute under +20 %.
```

**Expected**: topk3 IoU 0.13 → 0.30-0.40. Hand L2 24.5 → ~15 cm.
Foot L2 42.7 → ~22 cm.

**Reasoning chain**:

1. v8.1 already proved the loss + GT + supervision are correct
   (target <5cm hit 4.5 % → 11.6 %; topk3 IoU climbs from
   indistinguishable from chance to 2× chance).
2. The single-shot attention from per-part query to 128 object tokens
   is the **architectural bottleneck**. The query has no opportunity
   to refine its prediction, and no opportunity to coordinate with
   other body-part queries.
3. InteractVLM's CVPR 2025 result is **direct evidence** that
   mask-decoder-style heads (SAM decoder family) lift body contact
   F1 by 20+ pp absolute over previous SOTA (DECO). The architecture
   class is the right one.
4. Implementation effort: ~200 lines, +6 M params. Cheaper than
   re-extracting pseudo-labels at higher resolution; cheaper than
   adding a separate motion-kinematics dataloader.

**If this fails** (topk3 IoU < 0.25 after retrain), the next
follow-up is the **EgoChoir-style motion-KV stream**, which is
complementary and can stack on top. Combined motion-KV + mask-decoder
is the conjectured sweet spot for our setting.

## 6. Implementation file plan (v9)

| file | change | lines |
|---|---|---:|
| `src/piano/models/interaction_predictor.py` | new `MaskDecoder` (6 layers self-attn + cross-attn + FFN), replace v8.1 `CrossAttentionWeightsOnly` | +180 |
| `src/piano/training/losses.py` | auxiliary deep-supervision losses (loss at every decoder layer) | +30 |
| `src/piano/training/train_predictor.py` | accumulate aux losses with weight schedule | +20 |
| `tests/test_mask_decoder.py` | shape tests, gradient flow, deep-supervision contribution | +120 |
| `configs/training/predictor_v9_maskdecoder.yaml` | new config | +180 |

Total: ~530 lines.

If v9 hits gate, **v10 backlog** (queued, not yet decided):

- Motion-KV stream (EgoChoir-style with body-joint kinematics from
  `joint_xyz_per_frame`). Estimated +1.5 M params, +100 lines, +5 %
  training time.
- SkateFormer-style joint-partitioned self-attn within the trunk
  (replaces per-frame full-self-attn with physical-joint-partitioned
  attn). Estimated +0 M params (replaces existing), +250 lines.

## 7. Acceptance gates for v9

| metric | v8.1 best | **v9 target** | gate |
|---|---:|---:|---|
| topk3 mean IoU (primary) | 0.141 | **≥ 0.30** | hard |
| topk3 mean F1 | 0.184 | ≥ 0.42 | hard |
| target <5cm hit | 11.6 % | ≥ 18 % | hard |
| target <10cm hit | 26.9 % | ≥ 35 % | hard |
| pelvis pct<10cm | 59.1 % | ≥ 55 % | hard (no regress) |
| hand L2 (cm) | 24.5 | ≤ 19 | hard |
| **foot L2 (cm)** | **42.7** | **≤ 25** | hard (FIX) |
| phase macro F1 | 0.637 | ≥ 0.62 | soft |
| support macro F1 | 0.393 | ≥ 0.39 | soft |

Pass condition: 6/8 hard gates + foot fix retained.

If v9 fails:
- topk3 IoU < 0.25 → mask-decoder is not enough; add motion-KV (v10).
- foot L2 still > 30 cm → kinematic prior is the dominant issue;
  motion-KV is mandatory.
- topk3 IoU > 0.30 but pelvis regresses → tune deep-supervision
  weights or per-frame self-attn budget.

## 8. References (full citations)

### Direct architectural precedents

1. Dwivedi, S.K., Antić, D., Tripathi, S., Taheri, O., Schmid, C.,
   Black, M.J., Tzionas, D. **"InteractVLM: 3D Interaction Reasoning
   from 2D Foundational Models."** CVPR 2025. arXiv:2504.05303.
   Code: github.com/saidwivedi/InteractVLM.
2. Schult, J., Engelmann, F., Hermans, A., Litany, O., Tang, S.,
   Leibe, B. **"Mask3D: Mask Transformer for 3D Semantic Instance
   Segmentation."** ICRA 2023. arXiv:2210.03105.
   Code: github.com/JonasSchult/Mask3D.
3. Kolodiazhnyi, M., Vorontsova, A., Konushin, A., Rukhovich, D.
   **"OneFormer3D: One Transformer for Unified Point Cloud
   Segmentation."** CVPR 2024. arXiv:2311.14405. Code:
   github.com/oneformer3d/oneformer3d.
4. Yang, Y., Hou, K., Zhao, W., Zhu, X., Yan, S., Yang, S.
   **"EgoChoir: Capturing 3D Human-Object Interaction Regions from
   Egocentric Views."** NeurIPS 2024. arXiv:2405.13659.
   Code: github.com/yyvhang/EgoChoir_release.
5. Tripathi, S., Chatterjee, A., Passy, J.-C., Yi, H., Tzionas, D.,
   Black, M.J. **"DECO: Dense Estimation of 3D Human-Scene Contact In
   The Wild."** ICCV 2023. arXiv:2309.15273.
   Code: github.com/sha2nkt/deco.

### Motion-stream / kinematic encoder

6. Mehraban, S., Adeli, V., Taati, B. **"MotionAGFormer: Enhancing 3D
   Human Pose Estimation with a Transformer-GCNFormer Network."**
   WACV 2024. arXiv:2310.16288. Code:
   github.com/TaatiTeam/MotionAGFormer.
7. Do, J., Kim, M.G. **"SkateFormer: Skeletal-Temporal Transformer for
   Human Action Recognition."** ECCV 2024. arXiv:2403.09508. Code:
   github.com/KAIST-VICLab/SkateFormer.
8. Zhu, W., Ma, X., Liu, Z., Liu, L., Wu, W., Wang, Y. **"MotionBERT:
   A Unified Perspective on Learning Human Motion Representations."**
   ICCV 2023. arXiv:2210.06551. Code: github.com/Walter0807/MotionBERT.

### Encoder evidence (against upgrade)

9. Heuken, B., Goebbels, S., Pomerleau, F., Roth, M.
   **"Interpretable Affordance Detection on 3D Point Clouds with
   Probabilistic Prototypes."** arXiv:2504.18355 (May 2025; submitted
   to BMVC). Direct PointNet++ vs PT V3 ablation on 3D AffordanceNet,
   showing PT V3 loses 8.7 mIoU points.
10. Wu, X., Jiang, L., Wang, P.-S., Liu, Z., Liu, X., Qiao, Y., Ouyang,
    W., He, T., Zhao, H. **"Point Transformer V3: Simpler, Faster,
    Stronger."** CVPR 2024 (Oral). arXiv:2312.10035. Code:
    github.com/Pointcept/PointTransformerV3.
11. Wu, X., DeTone, D., Frost, D., Shen, T., Xie, C., Yang, N.,
    Engel, J., Newcombe, R., Zhao, H., Straub, J. **"Sonata:
    Self-Supervised Learning of Reliable Point Representations."**
    CVPR 2025 (Highlight). arXiv:2503.16429. Code:
    github.com/facebookresearch/sonata.
12. Zhou, J., Wang, J., Ma, B., Liu, Y.-S., Huang, T., Wang, X.
    **"Uni3D: Exploring Unified 3D Representation at Scale."** ICLR
    2024 (Spotlight). arXiv:2310.06773. Code:
    github.com/baaivision/Uni3D.
13. Qian, G., Li, Y., Peng, H., Mai, J., Hammoud, H., Elhoseiny, M.,
    Ghanem, B. **"PointNeXt: Revisiting PointNet++ with Improved
    Training and Scaling Strategies."** NeurIPS 2022. arXiv:2206.04670.

### Trunk capacity / scaling

14. Peebles, W., Xie, S. **"Scalable Diffusion Models with
    Transformers."** ICCV 2023 (Oral). arXiv:2212.09748. Code:
    github.com/facebookresearch/DiT.

### HOI affordance baselines (existing in v1 survey)

15. Peng, X., Xie, Y., Wu, Z., Jampani, V., Sun, D., Jiang, H.
    **"HOI-Diff: Text-Driven Synthesis of 3D Human-Object Interactions
    using Diffusion Models."** CVPR 2025 HuMoGen Workshop.
    arXiv:2312.06553. Code: github.com/neu-vi/HOI-Diff.
16. Cha, J., Kim, J., Yoon, J.S., Baek, S. **"Text2HOI: Text-guided
    3D Motion Generation for Hand-Object Interaction."** CVPR 2024.
    arXiv:2404.00562. Code: github.com/JunukCha/Text2HOI.

## 9. Companion analyses

- `analyses/2026-05-02_hoi_affordance_sota_survey_post_move_as_you_say.md` —
  v1 SOTA survey, 7 candidates with focus on output representation
  consensus (per-vertex / per-token / per-FPS-point / sparse joints).
- `analyses/2026-05-05_v8_round1_diagnosis_and_v81_plan.md` — v8.1
  plan (multi-hot binary GT + MoMask masking + drop consistency).
- `analyses/2026-05-05_v81_results_and_v811_plan.md` — v8.1 results
  (target <5cm hit 4.5 % → 11.6 %; foot L2 regressed; topk metric
  fix in v8.1.1).

## 10. Open questions for follow-up

1. EgoChoir's exact gradient-modulation equation for τ_v / τ_m —
   need to read the full PDF to confirm whether it's a multiplicative
   gate on the attention output or a scalar applied to gradients in
   backward. If the latter, the "modulation token" implementation is
   non-trivial (custom autograd hooks).
2. InteractVLM's mask-decoder is SAM (not SAM2) — does an SAM2
   port help, or does the temporal modeling not help for our 196-frame
   per-clip setup where we already have full self-attn over time?
3. Mask3D's auxiliary losses use Hungarian matching for a set of
   instance queries; in our case the queries are deterministic (T × P
   indices), so no matching needed. But the **deep-supervision loss
   at every decoder layer** is the load-bearing trick, and that
   carries over directly.
4. Is per-frame self-attn over P=5 body-part queries enough, or do
   we need T*P=980-query self-attn? Compute budget says probably the
   former. The MotionAGFormer pattern (parallel temporal-attn +
   spatial-attn) might be the right answer here.
