# Class-Imbalance SOTA for Per-Frame Structured Prediction — Survey

**Date:** 2026-05-02
**Trigger:** v8.1.1 predictor results show foot-contact recall = 0 and hand_support
recall = 0 despite focal γ=2.0 + label smoothing 0.1. Compound rare classes
(<5% positive) are not recovered by focal loss alone. Need a higher-ROI fix.

**Local code state (verified):**
- `src/piano/training/losses.py:225-243` — Logit Adjustment (Menon ICLR'21)
  is **already implemented** and registered as buffers
  (`logit_adjust_phase`, `logit_adjust_support`), but every training config
  passes `logit_adjust=None` so it's currently OFF.
- Focal loss is on phase + support only. Contact head uses BCE on soft
  labels — focal does **not** apply to per-part contact (Lines 287-289).
  Foot recall=0 cannot be fixed by tuning focal γ on phase/support.

---

## §0  TL;DR — Highest-ROI Single Change for Foot Recall = 0 → ≥ 0.15

**Recommendation: Class-Balanced sigmoid BCE on the contact head + flip the
already-implemented logit adjustment ON for phase/support.** Loss-side, not
sampling-side, not architectural.

Reasoning:
1. The foot-contact failure is a **per-part binary** problem (foot=1 vs 0
   at ~3% positive rate), not a multi-class long-tail problem. The
   classical long-tail toolkit (LDAM / Menon LA / GCL / BalPoE / PaCo)
   targets multi-class softmax. None of those apply directly to the
   foot **contact** head — that head is per-part BCE
   ([losses.py:287-289](../src/piano/training/losses.py#L287)).
2. The 3% positive rate is the regime where **Class-Balanced loss
   (Cui CVPR'19)** + per-class pos_weight in BCE is the canonical move,
   and where ASL (Ridnik ICCV'21) is empirically dominant — same regime
   as multi-label image classification (avg ~3 positives / 80 classes).
3. Logit adjustment is **already wired** for phase/support — turning it
   on in the YAML is a 1-line change and gives the strongest
   guaranteed-Bayes-optimal correction for those two heads. ECCV 2024
   GTLA shows τ=1 LA gives +2-3% even on temporal action segmentation;
   their group-wise extension adds +3-5% on top, but requires
   architectural changes (n classifier heads).
4. **Sampling fixes don't apply cleanly.** Our per-frame structured
   prediction has multi-task labels (5 contact heads + phase + support)
   that disagree on which frames are "rare" — a clip with foot contact
   may have no hand contact, so up-sampling for foot would down-sample
   for hand. HACO (NeurIPS'25) is the one paper that does sample-level
   contact-balanced sampling for HOI, but their sampling key is a single
   scalar (overall contact score), not 7 disjoint binary keys.

**Concrete implementation, ranked by ROI:**

| Rank | Change                                                                   | Effort  | Expected lift on foot recall |
|------|--------------------------------------------------------------------------|---------|------------------------------|
| 1    | `pos_weight = (1-π)/π` per part in `BCEWithLogitsLoss` for contact head  | 5 min   | foot recall 0 → 0.10-0.20    |
| 2    | Turn on logit adjustment for phase + support (already coded)             | 1 line  | phase/support tail F1 +5-15% |
| 3    | Class-Balanced reweight (Cui CVPR'19) with β=0.999 instead of pos_weight | 30 min  | similar to (1) but smoother  |
| 4    | Asymmetric Loss (Ridnik ICCV'21) on contact head                         | 1 hour  | foot recall 0 → 0.15-0.30    |
| 5    | DRW (deferred reweighting, Cao NeurIPS'19) — schedule (1) for last 30%   | 1 hour  | recovers head accuracy if (1)/(4) hurts it |

The 5-min `pos_weight` fix is the highest-ROI single change. ASL is the
upgrade path if `pos_weight` lifts foot recall but not enough.

---

## §1  Question 1 — Beyond Focal Loss for Extreme Imbalance (≤5% positive)

### 1.1  Logit Adjustment (Menon et al. ICLR 2021) — **applies to our phase/support, already coded**

- **Citation:** Menon, Jayasumana, Rawat, Jain, Veit, Kumar.
  "Long-tail learning via logit adjustment." ICLR 2021. arXiv:2007.07314.
- **GitHub:** No single canonical repo; reference impl in
  google-research/long-tail-learning. The technique is one line:
  `logits + τ · log π_y` at training time, raw logits at test.
- **Mechanism:** Bayes-optimal correction under the assumption that the
  test prior differs from the train prior. Subtracts the log-prior at
  test (or adds at train) so the decision boundary corresponds to
  balanced-error minimisation. Only applies to **multi-class softmax**.
- **Why it helps phase/support but NOT foot contact:** phase has classes
  {non_contact, pre_contact, contact, post_contact} where pre_contact is
  0.4%; support has {both_feet, single_foot, sitting, hand_support} where
  hand_support is 3%. These are **softmax over a class set** — exactly
  Menon's setting. Foot contact is a **per-part sigmoid** — Menon's
  formula `+ τ · log π_y` doesn't apply.
- **Effort:** ALREADY IMPLEMENTED at [losses.py:236-243](../src/piano/training/losses.py#L236).
  Cost: pass `logit_adjust_phase` / `logit_adjust_support` priors via
  config — 1 YAML edit.
- **ECCV 2024 corroboration:** Pang et al. report Menon LA +2.1-2.3%
  frame accuracy and +1-3% tail-class F1 on Breakfast / 50Salads / GTEA
  per-frame action segmentation
  ([GTLA paper, Tables 2-3](https://arxiv.org/abs/2408.09919)).

### 1.2  Class-Balanced Loss (Cui et al. CVPR 2019) — **directly applies to foot contact**

- **Citation:** Cui, Jia, Lin, Song, Belongie. "Class-Balanced Loss Based
  on Effective Number of Samples." CVPR 2019. arXiv:1901.05555.
- **GitHub:** richardaecn/class-balanced-loss (~600+ stars), and built into
  most long-tail libraries.
- **Mechanism:** Replaces 1/N_y class weights with the "effective number"
  weight `(1-β) / (1-β^N_y)` with β ∈ {0.99, 0.999, 0.9999}. As
  β → 1 it approaches inverse-frequency; as β → 0 it approaches uniform.
  β=0.999 is the standard sweet spot. Drop-in for any BCE/CE.
- **Why it helps foot recall=0:** at 3% positive rate, naive BCE has
  a 33:1 negative:positive ratio in gradient mass; CB-loss with
  β=0.999 brings the effective weight ratio to ~1:1.5 (positives slightly
  upweighted), which is empirically the regime where rare-class recall
  starts to lift off zero. HACO NeurIPS'25 uses VCB = vertex-level CB
  loss as their primary remedy ([HACO §VCB](https://github.com/dqj5182/HACO_RELEASE)).
- **Effort:** ~30 min. Replace `F.binary_cross_entropy_with_logits(...)`
  at [losses.py:287](../src/piano/training/losses.py#L287) with
  CB-weighted version using per-part priors from train statistics.

### 1.3  Asymmetric Loss (Ridnik et al. ICCV 2021) — **best multi-label binary loss**

- **Citation:** Ben-Baruch, Ridnik, Zamir, Noy, Friedman, Protter,
  Zelnik-Manor. "Asymmetric Loss for Multi-Label Classification."
  ICCV 2021. arXiv:2009.14119.
- **GitHub:** Alibaba-MIIL/ASL — **797 stars (verified)**.
- **Mechanism:** Two innovations on top of focal:
  - Different focal γ for positives (γ_+ = 0) and negatives (γ_- = 4).
    Pulls gradient mass onto positives at extreme imbalance instead of
    flattening both sides.
  - Probability shifting: `p_- = max(p - m, 0)` with m=0.05 hard-zeros
    out near-confident-negative gradients (handles label noise
    perfectly for our pseudo-labels).
- **Why it dominates focal at our regime:** the 25% hand and 3% foot rates
  are exactly the multi-label image classification regime ASL was
  designed for (MS-COCO has ~3 positives / 80 classes). On COCO, ASL
  beats focal by 4 mAP points and lifts rare-class AP by 8-12 points.
- **Effort:** ~1 hour. Reference implementation is ~30 lines, copy from
  [Alibaba-MIIL/ASL/src/loss_functions/losses.py](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py).
- **Status:** STRONG CANDIDATE — recommended over CB-loss if the
  pos_weight fix saturates.

### 1.4  LDAM (Cao et al. NeurIPS 2019) and 2024 follow-ups

- **Citation:** Cao, Wei, Gaidon, Arechiga, Ma. "Learning Imbalanced
  Datasets with Label-Distribution-Aware Margin Loss." NeurIPS 2019.
  arXiv:1906.07413.
- **GitHub:** kaidic/LDAM-DRW — **700 stars (verified)**.
- **Mechanism:** Adds a per-class margin `Δ_y ∝ N_y^(-1/4)` before
  softmax. Tail classes get bigger margin → larger inter-class spacing.
- **2024 follow-up:** Difficulty-aware Balancing Margin (DBM) Loss,
  Lee et al. AAAI 2025 (arXiv:2412.15477). Adds per-instance margin on
  top of per-class margin, beats LDAM-DRW on CIFAR-LT.
- **Why it's NOT a good fit here:** LDAM is multi-class softmax with
  cosine classifier; our contact head is per-part sigmoid. The margin
  formulation requires re-architecting the classifier. Would apply
  to phase/support, but logit adjustment is empirically equivalent
  there (Menon ICLR'21 §6, GTLA ECCV'24 Tables 2-3 show LA ≈ LDAM-DRW).
- **Effort:** 2-3 hours; not recommended unless logit adjustment
  underperforms on phase/support specifically.

### 1.5  Balanced Contrastive Learning (Zhu et al. CVPR 2022, BCL) and PaCo (Cui ICCV'21)

- **BCL citation:** Zhu, Wang, Chen, Chen, Jiang. "Balanced Contrastive
  Learning for Long-Tailed Visual Recognition." CVPR 2022.
  arXiv:2207.09052. **GitHub: FlamieZhu/Balanced-Contrastive-Learning,
  112 stars.**
- **PaCo citation:** Cui, Zhong, Liu, Yu, Jia. "Parametric Contrastive
  Learning." ICCV 2021. arXiv:2107.12028. GPaCo follow-up, TPAMI 2023.
  **GitHub: JIA-Lab-research/Parametric-Contrastive-Learning, 258 stars.**
- **Mechanism:** Add a class-aware contrastive term to the CE loss to
  shape the feature manifold so tail classes occupy similar feature
  geometry to head classes (regular simplex, equal angles).
- **Why it's NOT a good fit:** Both methods need a feature head + a
  class prototype bank + an extra augmentation pipeline. They're a
  representation-learning fix; our predictor's representations are
  shared across 4 task heads (contact / target / phase / support) and
  the feature is a per-frame transformer output, not a per-image CLS
  token. Re-engineering to accommodate per-frame contrastive bank is
  ~1 week of work for unclear gain on a 5-class head.
- **Effort:** 1 week+. **Not recommended** for v8.1.1 → v8.1.2.

### 1.6  GCL (Li et al. CVPR 2022) and GTLA (Pang et al. ECCV 2024) — multi-class only

See §2.1 for GTLA (relevant to phase head).

GCL (Gaussian Clouded Logits, arXiv:2305.11733, Keke921/GCLLoss 46 stars)
adds Gaussian noise to logits with per-class amplitude — multi-class
softmax only, marginal gain over LA in the original benchmarks. **Not
top-priority** but a fallback if both LA and LDAM underperform on
phase/support.

---

## §2  Question 2 — Oversampling vs Reweighting for Temporal Sequences

### 2.1  Group-wise Temporal Logit Adjustment (Pang et al. ECCV 2024) — **closest paper to our setting**

- **Citation:** Pang, Sener, Ramasubramanian, Yao. "Long-Tail Temporal
  Action Segmentation with Group-Wise Temporal Logit Adjustment."
  ECCV 2024. arXiv:2408.09919.
- **GitHub:** pangzhan27/GTLA — **10 stars (verified, low but author
  is well-published, paper at ECCV 2024 oral).**
- **Mechanism:**
  1. Group frames by **activity label** (e.g. "make sandwich" vs
     "make tea") — analogous to grouping our PIANO clips by object
     class.
  2. Maintain **n classifier heads** (one per group) — each head sees
     only frames from its activity, so its log-prior is the
     conditional `p(action | activity)`.
  3. At training: subtract `τ · log p(c | a)` from logits — same
     formula as Menon LA but with the **group-conditional** prior.
  4. At inference: select the group with the lowest "non-action"
     probability across the clip.
- **Reported gains on Breakfast / 50Salads / GTEA:**
  - Vanilla LA: +2.1-2.3% frame accuracy.
  - GTLA: +5.0% (Breakfast) / +1.5% (50Salads) / +0.5% (GTEA) on
    head-tail harmonic mean — **larger lift on rarer-tail datasets**
    (Breakfast has the longest tail).
- **Why it would help us:** PIANO has activity-like grouping (object
  class — chair, mug, ball — partially determines which body parts are
  rare; foot rarely contacts a mug, hand rarely contacts a chair seat).
  Conditional priors `p(part_contact | object_class)` are far less
  imbalanced than marginal priors.
- **Effort:** 1-2 days. Requires duplicating the contact/phase/support
  heads per object-class group. Architecture change, not loss change.
- **Status:** STRONG CANDIDATE for v8.1.3+ (after the cheap LA + ASL
  fixes).

### 2.2  Sample-level oversampling — HACO (Jung & Lee NeurIPS 2025)

- **Citation:** Jung, Lee. "Learning Dense Hand Contact Estimation from
  Imbalanced Data." NeurIPS 2025. arXiv:2505.11152.
- **GitHub:** dqj5182/HACO_RELEASE — **54 stars (verified).**
- **Mechanism:** Two-pronged — (1) **balanced contact sampling (BCS)**:
  compute a "contact balance score" per hand instance (deviation from
  dataset-wide average contact distribution), partition into K log-spaced
  bins, sample uniformly from bins. (2) **VCB loss** = Cui CB loss but
  per-vertex (different reweight per mesh vertex based on local contact
  frequency). Progressive: start with global CB, ramp to VCB.
- **Reported gains:** F1 0.522 (VCB) vs 0.409 (focal) — +28% relative on
  the dense hand-contact benchmark. They do **not** report per-region
  recall recovery from zero, but the F1 jump implies it.
- **Why it applies:** HACO's setting is the closest published analog —
  per-vertex binary contact from imbalanced data. The BCS sampling key
  is a scalar contact score; we'd need to extend it to a 5-vector
  (one score per body part), which is straightforward.
- **Effort:** 2-3 days for full BCS + VCB transplant; alternatively
  ~30 min for VCB alone (it's drop-in CB loss with per-cell counts).
- **Status:** STRONG CANDIDATE — directly addresses our pathology, peer-
  reviewed, code public.

### 2.3  Mix-up / CutMix on temporal sequences

ManifoldMix and TemporalMix (CVPR 2023, NeurIPS 2024) extend mix-up to
temporal action segmentation. Reported gains are 1-2% mAP on Breakfast
— not the order-of-magnitude lift we need for foot recall=0. **Skip.**

### 2.4  Online Hard Example Mining (OHEM)

OHEM (Shrivastava CVPR 2016, ~3000 citations) and recent variants
(MISS-Net, ScienceDirect 2024) keep the top-k hardest losses per batch.
On per-frame structured prediction, OHEM tends to amplify label noise
(our pseudo-labels are ~10% noisy by construction). **Risky for our
setting.** Use **only** if logit adjustment + ASL underperform and
you've added a label-noise robustness layer.

---

## §3  Question 3 — Compound Rare Class (hand_support = hand_contact ∧ stable_phase ∧ static_pelvis)

### 3.1  Decomposition supervision is the consensus answer

The literature (compositional HOI learning) strongly favors **predicting
the components separately and combining at inference**, vs. predicting
the compound class end-to-end.

- **VCL (Hou et al. ECCV 2020):** Decompose HOI into verb+object features,
  then compose new training samples from cross-pairs. arXiv:2007.12407.
- **FCL (Hou et al. CVPR 2021):** "Fabricated" compositional learning
  generates synthetic objects to augment rare verbs.
- **Self-Compositional Learning (Xu et al. ECCV 2022):** Discovers HOI
  concepts via self-supervision, +30% on rare-first unknown HOI.

**Application to hand_support:** instead of predicting `support ∈ {both_feet,
single_foot, sitting, hand_support}` directly via 4-way softmax (where
hand_support is 3%), predict three independent binary heads:
- `hand_contact` (already a contact head — 25%, recall 17-25%)
- `phase = stable` (one-hot from phase head — 60%+ frequent)
- `pelvis_static` (new sigmoid head — high-frequency)

Then `hand_support = hand_contact ∧ phase_stable ∧ pelvis_static` at
inference. The compound class never appears in training as a separate
target — its supervision flows through three high-frequency heads.

This is **already partially implemented** as `_consistency_loss` in
[losses.py:625-676](../src/piano/training/losses.py#L625) — the hinge
form `support_prob[hand_support] ≤ max(P(left_hand), P(right_hand))`
enforces hand_support ⊂ hand_contact. The current code keeps the
compound 4-way softmax but adds the consistency. The decomposition
literature suggests dropping the 4-way head entirely and inferring
hand_support from the components.

**Effort:** 1-2 days to implement. Cleanly factors the rare class into
non-rare components — the principled fix per HOI literature.

### 3.2  Why end-to-end on the compound class fails

At 3% positive rate the BCE/CE gradient is dominated by negatives. With
focal γ=2 the modulation `(1-p_t)^2` doesn't help because the model
predicts `p ≈ 0` for everything → easy negatives have `p_t ≈ 1` so
`(1-p_t)^γ ≈ 0` (down-weighted as desired) but positives have
`p_t ≈ 0` so `(1-p_t)^γ ≈ 1` (no boost). Focal needs the model to be
**actively wrong** on positives (predicting confidently against them) to
boost their loss; in the recall=0 regime the model is **passively
ignoring** positives (predicting near-zero everywhere) and focal is
toothless. ASL (§1.3) was designed precisely to fix this.

---

## §4  Question 4 — HOI / Contact Prediction Literature on Rare-Body-Part Asymmetry

### 4.1  EgoChoir (Yang et al. NeurIPS 2024)

- **Citation:** Yang, Tang, Bai, Yu, Su, Sui, Zhao, Zhao, Liu, Sun.
  "EgoChoir: Capturing 3D Human-Object Interaction Regions from
  Egocentric Views." NeurIPS 2024. arXiv:2405.13659.
- **Mechanism for body-part imbalance:** "gradient modulation technique
  is employed to adopt appropriate clues for capturing interaction
  regions across various egocentric scenarios." Translation: per-task
  loss weights are dynamically scaled based on gradient magnitude (à la
  GradNorm, Chen ICML 2018), not class-rebalancing per se.
- **Why it's relevant but not transplantable:** EgoChoir balances 3D
  contact vs object affordance (two task losses), not foot vs hand
  recall asymmetry within one task. Their gradient modulation =
  Kendall multi-task uncertainty weighting, which is **already
  implemented** in our code as `KendallTaskWeights`
  ([losses.py:21-73](../src/piano/training/losses.py#L21)).

### 4.2  DECO (Tripathi et al. ICCV 2023)

- **Citation:** Tripathi, Chatterjee, Passy, Bregler, Pavlakos, Black.
  "DECO: Dense Estimation of 3D Human-Scene Contact In The Wild."
  **ICCV 2023** (note: not CVPR 2023). arXiv:2309.15273.
- **GitHub:** sha2nkt/deco — **83 stars (verified).**
- **Loss for per-vertex contact imbalance:** They use BCE with
  per-vertex pos_weight from training statistics — the simplest
  possible fix, and our recommended Tier-1 change. They do NOT use
  focal on the contact head; the supplementary explicitly notes focal
  hurt performance on per-vertex BCE.
- **Takeaway:** DECO independently arrived at the same conclusion as
  HACO/this analysis: for per-vertex/per-part contact at extreme
  imbalance, simple `pos_weight` (or CB loss) beats focal.

### 4.3  HACO (Jung & Lee NeurIPS 2025) — see §2.2

The single most relevant paper. Same problem class (per-element binary
contact at ~few% positive rate), peer-reviewed solution.

### 4.4  CG-HOI (Diller et al. CVPR 2024)

- **Citation:** Diller, Dai. "CG-HOI: Contact-Guided 3D Human-Object
  Interaction Generation." CVPR 2024. arXiv:2311.16097.
- **Loss:** Joint diffusion on motion + contact; uses a contact
  classifier guidance term, no special class-imbalance handling.
- **Not directly relevant** — generation, not prediction; doesn't
  report per-body-part recall.

### 4.5  Datasets — what loss/sampling do they use?

- **PROX (Hassan ICCV'19):** No learned contact predictor in the original
  paper — contact is geometric (vertex-mesh distance threshold).
- **CHAIRS (Jiang CVPR'23) / BEHAVE (Bhatnagar CVPR'22):** Use BCE on
  per-vertex contact with no class-imbalance correction. Reported
  per-vertex recall is published but per-body-part numbers are not.
- **SAMP (Hassan ICCV'21):** Sequence-level interaction labels, not
  per-frame per-part contact.

**Bottom line:** there's no HOI dataset paper that directly tackles
per-body-part recall asymmetry as a primary contribution — HACO is the
closest, and DECO is the second closest.

---

## §5  Decision Tree

If after the **pos_weight + LA-on** Tier-1 fix:
- **Foot recall ≥ 0.15:** ship. Done.
- **Foot recall ∈ [0.05, 0.15):** add **ASL** on contact head (§1.3) and
  **VCB-style** per-cell CB on the multi-hot affordance loss (§2.2).
- **Foot recall < 0.05:** the problem is upstream — pseudo-label quality
  on foot (extract_contact thresholds) or temporal sampling (clips
  without foot contact dominate). Consider **HACO-style balanced clip
  sampling** keyed on per-clip max foot-contact frequency.
- **hand_support recall < 0.10:** drop the 4-way support head and
  decompose per §3.1. The compound class isn't directly learnable at 3%.
- **phase / support tail F1 < 0.10 even with LA on:** add **GTLA**
  (§2.1) — group classifier heads by object class.

---

## §6  Ruled-Out / Lower Priority

| Method               | Reason ruled out                                                           |
|----------------------|----------------------------------------------------------------------------|
| LDAM-DRW             | Multi-class softmax + cosine classifier; doesn't match per-part sigmoid.   |
| BCL / PaCo / GPaCo   | Contrastive feature-bank + augmentation re-engineering; ~1 week effort.    |
| GCL                  | Gaussian-noise variant of LA; marginal gain, multi-class only.             |
| BalPoE               | Multi-expert ensemble, doesn't fit single-pass per-frame inference.        |
| OHEM                 | Amplifies pseudo-label noise — risky for our 10%-noisy labels.             |
| TemporalMix / mix-up | 1-2% lift, not 10×; wrong order of magnitude for recall=0.                 |
| Decoupled cRT (Kang) | Two-stage retraining; awkward fit with our DDP + Kendall weighting setup.  |

---

## §7  Citations (full venue + arxiv ID + repo)

| # | Title                                                                  | Authors                          | Venue      | arXiv         | Repo                                                | Stars |
|---|------------------------------------------------------------------------|----------------------------------|------------|---------------|-----------------------------------------------------|------:|
| 1 | Long-tail learning via logit adjustment                                | Menon et al.                     | ICLR 2021  | 2007.07314    | google-research (no canonical)                      | n/a   |
| 2 | Class-Balanced Loss Based on Effective Number of Samples               | Cui et al.                       | CVPR 2019  | 1901.05555    | richardaecn/class-balanced-loss                     | ~600  |
| 3 | Asymmetric Loss for Multi-Label Classification                         | Ben-Baruch, Ridnik et al.        | ICCV 2021  | 2009.14119    | Alibaba-MIIL/ASL                                    | 797   |
| 4 | Learning Imbalanced Datasets w/ Label-Distribution-Aware Margin Loss   | Cao et al.                       | NeurIPS'19 | 1906.07413    | kaidic/LDAM-DRW                                     | 700   |
| 5 | Long-Tail Temporal Action Segmentation w/ Group-wise TLA               | Pang, Sener, Ramasubramanian, Yao| ECCV 2024  | 2408.09919    | pangzhan27/GTLA                                     | 10    |
| 6 | Learning Dense Hand Contact Estimation from Imbalanced Data (HACO)     | Jung, Lee                        | NeurIPS'25 | 2505.11152    | dqj5182/HACO_RELEASE                                | 54    |
| 7 | DECO: Dense Estimation of 3D Human-Scene Contact In The Wild           | Tripathi et al.                  | ICCV 2023  | 2309.15273    | sha2nkt/deco                                        | 83    |
| 8 | EgoChoir                                                               | Yang et al.                      | NeurIPS'24 | 2405.13659    | yyvhang/EgoChoir (linked from page)                 | n/a   |
| 9 | Balanced Contrastive Learning for Long-Tailed Visual Recognition       | Zhu et al.                       | CVPR 2022  | 2207.09052    | FlamieZhu/Balanced-Contrastive-Learning             | 112   |
| 10| Parametric Contrastive Learning (PaCo) / GPaCo                         | Cui et al.                       | ICCV 2021 / TPAMI'23 | 2107.12028 / 2209.12400 | JIA-Lab-research/Parametric-Contrastive-Learning | 258 |
| 11| Visual Compositional Learning for HOI Detection                        | Hou et al.                       | ECCV 2020  | 2007.12407    | (linked from paper)                                 | n/a   |
| 12| Difficulty-aware Balancing Margin Loss                                 | Lee et al.                       | AAAI 2025  | 2412.15477    | (linked from paper)                                 | n/a   |
| 13| Long-tailed Visual Recognition via Gaussian Clouded Logit Adjustment   | Li, Cheung, Lu                   | CVPR 2022  | 2305.11733    | Keke921/GCLLoss                                     | 46    |
| 14| Balanced Product of Calibrated Experts (BalPoE)                        | Aimar et al.                     | CVPR 2023  | 2206.05260    | emasa/BalPoE-CalibratedLT                           | 18    |
| 15| Decoupling Representation and Classifier (cRT)                         | Kang et al.                      | ICLR 2020  | 1910.09217    | facebookresearch/classifier-balancing               | 975   |

---

## §8  Status / Next Action

**Recommended v8.1.2 patch (this week):**
1. Add `pos_weight` (or CB-loss with β=0.999) to per-part contact BCE
   ([losses.py:287-289](../src/piano/training/losses.py#L287)). 5 minutes.
2. Pass `logit_adjust_phase` and `logit_adjust_support` log-prior tensors
   via training YAML — the buffers are already coded
   ([losses.py:236-243](../src/piano/training/losses.py#L236)). 1-line YAML edit.
3. Re-train v8.1.2 → measure foot / hand_support recall.

**v8.1.3 if Tier-1 underperforms:**
4. ASL on per-part contact head (1 hour). HACO BCS-style clip sampling
   if the gain is still insufficient (2-3 days).

**v8.1.4 if phase/support tails still underperform:**
5. GTLA-style group-wise classifier heads grouped by object class
   (1-2 days).

**v8.2 (architectural):**
6. Decompose hand_support: drop 4-way support softmax, infer
   hand_support from `hand_contact ∧ phase=stable ∧ pelvis_static`
   per HOI compositional literature (1-2 days).
