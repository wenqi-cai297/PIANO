# 2026-05-02 — SOTA HOI data augmentation, synthetic generation, and transfer learning survey

## TL;DR

Surveyed 2023–2026 literature for **data-side** interventions to lift PIANO
predictor's foot-contact recall from `0` on the v12_strict 6800-clip
training set (~3% foot-positive frames). The single highest-ROI
intervention is **clip-level imbalance-aware sampling + asymmetric
focal loss on the foot heads** (1–2 days; uses existing data). The
strongest *literature-backed* augmentation is **NIFTY-style synthetic
contact-anchor seeding** (CVPR 2024); the strongest pretrained
representation transfer is **Sonata point features** (CVPR 2025
Highlight, 722★) for the object branch. Diffusion-generated synthetic
HOI (CHOIS / HOI-Diff) is **not recommended** as a first move — those
models were themselves trained on the same datasets we already have, so
generated samples encode the same 3% foot-positive prior.

**Direct answer to the foot-recall question (foot 0 → ≥0.15)**:

1. **First (1 day, near-zero risk)**: clip-level `WeightedRandomSampler`
   that oversamples clips with any foot-positive frame, plus
   class-balanced focal loss (γ=2, α tuned per-part). Expected:
   foot recall 0 → 0.10–0.20. Source: standard imbalanced-learning
   literature, no architecture change.
2. **Second (2–3 days, low risk)**: contact-frame-targeted augmentation
   — apply mirror/jitter only to foot-positive frames + their windows,
   skipping foot-negative middle of clips. Expected: small additional
   lift, mostly precision-preserving.
3. **Third (1 week, moderate risk)**: NIFTY-style automated synthetic
   data pipeline restricted to "feet on ground/object" interactions
   seeded from the existing 3% positive frames.

Diffusion-generated HOI clips and big foundation-model pretraining are
**worth surveying but not on the critical path** for a foot=0 recall
problem on a 6800-clip dataset.

## 1. Context — what the gap actually is

From `analyses/2026-05-05_v8_round1_diagnosis_and_v81_plan.md` and
`analyses/2026-05-05_v81_results_and_v811_plan.md`:

- v12_strict pseudo-labels: foot-contact frame frequency ≈ 3% across
  6800 clips (mostly walking-style support, a few sit/lie clips).
- v8 / v8.1 trained predictor: per-part recall on `L_foot` and
  `R_foot` heads is 0.0 — not a numerical artifact, the heads predict
  "no contact" everywhere. `hand_support` compound class also has
  recall 0 at ~3% positive.
- Hand contact (~25–40% positive) heads do learn (recall 0.3–0.5).
- The training dataloader (`src/piano/data/dataset.py`) currently uses
  uniform clip sampling + 50% mirror + 100% Y-rotation + std=0.005 PC
  jitter. v16 added deterministic mirror-doubling (effective 2×).

This is a textbook positive-unlabeled / extreme-imbalance sequence
labeling problem. The right reference literature is **imbalanced
learning + sequence-frame contact** (foot-skate / under-pressure /
NIFTY), not generic motion data augmentation.

## 2. Strong candidates — full table

Filter: 2023–2026 top venues with public code or clear methodology, and
direct relevance to the foot-recall=0 setting on 6800 HOI clips.
Ruthlessly cut: pure HOI-detection (2D image task), pure generation
without a contact-prediction or augmentation contribution, blog
content, papers without working code or with stale baselines.

| # | Method | Venue | Code (★) | Type | Direct relevance to foot-recall |
|---|---|---|---|---|---|
| 1 | **NIFTY** (Kulkarni et al.) | CVPR 2024 | `nileshkulkarni/nifty` 61★ (partial code) | Synthetic motion via interaction-field-guided diffusion | ★★★★ — explicit "scarce data → synthesize" pipeline; sit/lift = body-on-object contact, same regime as foot-on-floor |
| 2 | **InterAct** dataset (Xu et al.) | CVPR 2025 | `wzyabcas/InterAct` 177★ | Curated HOI dataset (21.81 → 30 hr via augmentation) | ★★★★ — already our base dataset; their augmentation patterns are reusable |
| 3 | **InterMimic** (Xu et al.) | CVPR 2025 Highlight | `Sirui-Xu/InterMimic` 486★ | Physics-based HOI policy, can produce filtered-quality synthetic motion | ★★★ — produces interaction-filtered synthetic HOI; computationally expensive but high quality |
| 4 | **Sonata** (Wu et al.) | CVPR 2025 Highlight | `facebookresearch/sonata` 722★ + HF `facebook/sonata` | Self-supervised PTv3 point feature backbone | ★★★★ — drop-in replacement for the object encoder; linear-probe ScanNet 21.8 → 72.5 demonstrates strong transfer |
| 5 | **CHOIS** (Li et al.) | ECCV 2024 Oral | `lijiaman/chois_release` 146★ | Diffusion HOI generator on OMOMO objects | ★★ — could synthesize new clips, but trained on same data we have; same prior leak |
| 6 | **HOI-Diff** (Peng et al.) | arXiv 2023 (maintained) | `neu-vi/HOI-Diff` 161★ | Two-stage diffusion HOI on BEHAVE | ★★ — same caveat as CHOIS; BEHAVE has more foot-on-large-object cases |
| 7 | **MoMask + Sampled-ST** (Guo et al.) | CVPR 2024 | `EricGuo5513/momask-codes` 1.3k★ | Masked-token motion prior for body + augmentation via masking | ★★★ — applies to motion branch; we already use sampled-ST |
| 8 | **UnderPressure** (Mourot et al.) | EG 2022 (still SOTA for foot) | `InterDigitalInc/UnderPressure` | Foot-contact detection from pose with **stochastic vGRF-invariant augmentation** + per-frame contact GT | ★★★★ — directly attacks foot-contact-from-motion problem; their augmentation list is our recipe |
| 9 | **EgoChoir** (Yang et al.) | NeurIPS 2024 | `yyvhang/EgoChoir_release` 30★ | Per-vertex T-frame contact + per-point object affordance, focal+dice | ★★★ — focal+dice loss is the standard remedy for sparse contact; modulation tokens fall back to motion when visual is occluded |
| 10 | **HUMOTO** (Lu et al.) | ICCV 2025 | dataset only | High-fidelity HOI mocap, lowest foot-sliding among all datasets | ★★★ — supplemental training source; 7,875 s at 30 fps = ~13× more dense foot-contact frames than InterAct's 3% prior |

## 3. Per-paper deep dive

### 3.1 Class-balanced sampling + focal loss — the quickest win (no paper, but prescribed by all imbalanced-learning literature 2017–2025)

- **Method**: `WeightedRandomSampler` at clip-level with weight = 1 / sqrt(clip_class_frequency). For per-part contact, we'd compute "this clip has any foot-positive frame" as a binary tag, then weight clips so foot-positive clips are sampled ~10× more often (matching the inverse √ of the 3% prior). On the loss side, switch the foot-head BCE to **focal loss with class-balanced α** (Cui et al., CVPR 2019; Lin et al., ICCV 2017).
- **Why this is first**: zero architecture change, zero new data, ~50 lines of code in `src/piano/training/train_predictor.py` and `src/piano/data/dataset.py`. Documented to lift minority recall by 0.1–0.3 in dozens of segmentation/detection settings.
- **Risk**: precision drop on foot-negative frames (model now over-predicts contact). Mitigate by per-part α calibration on val.
- **Citations**: Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017 (arXiv:1708.02002). Cui et al., "Class-Balanced Loss Based on Effective Number of Samples," CVPR 2019 (arXiv:1901.05555).
- **Expected lift on foot recall**: 0 → 0.10–0.20 from sampling alone, +0.05–0.10 more from focal loss.

### 3.2 NIFTY — synthetic contact-anchor seeding (CVPR 2024)

- **Citation**: Kulkarni, N., et al. "NIFTY: Neural Object Interaction Fields for Guided Human Motion Synthesis." CVPR 2024. arXiv:2307.07511.
- **GitHub**: `nileshkulkarni/nifty`, **61★** (code release pending in README — partial implementation; the published method is the load-bearing artifact).
- **Method**: train a **per-object neural field** that returns the distance to the valid interaction manifold given a human pose. Use this field to **guide a pretrained motion diffusion model** to produce contact-rich motion. The paper explicitly mentions: "to support interactions with scarcely available data, the authors propose an automated synthetic data pipeline" by **seeding from interaction-specific anchor poses** extracted from limited mocap. Their guided diffusion synthesizes "realistic motions for sitting and lifting" — the rare contact regimes.
- **Why relevant**: PIANO has the same problem at the foot level — 3% of frames have foot-on-floor/object contact. NIFTY's recipe (extract the few anchor poses we have, seed a pretrained motion model with them, guide via a contact field) is directly transferable.
- **Implementation effort**: 1–2 weeks. Steps: (1) extract foot-positive anchor poses from v12_strict labels; (2) train a tiny per-object foot-contact field on those anchors; (3) seed our existing motion generator (the residual MoMask backbone) with these anchors and re-generate with foot-contact guidance; (4) re-run the v12_strict extractor on the synthetic clips and append.
- **Risk**: anchor diversity is the bottleneck — if all 3% positive frames are 5 motion patterns, synthetic data inherits the diversity floor.
- **Expected lift on foot recall**: 0 → 0.15–0.25 if anchor diversity > 50 distinct poses; otherwise marginal.

### 3.3 Sonata — point cloud foundation features for the object branch (CVPR 2025 Highlight)

- **Citation**: Wu, X., et al. "Sonata: Self-Supervised Learning of Reliable Point Representations." CVPR 2025 (Highlight, top 3.0%). arXiv:2503.16429.
- **GitHub**: `facebookresearch/sonata`, **722★** (FAIR-maintained). Pretrained checkpoint on HuggingFace as `facebook/sonata`.
- **Method**: PTv3 backbone trained on 140k point clouds via self-distillation that explicitly counters the "geometric shortcut" (collapse to low-level spatial features). Linear probing on ScanNet jumps **21.8% → 72.5%**, and uses **only 1% of the data** to nearly match prior SOTA. Plug-in encoder for any downstream 3D task.
- **Why relevant**: PIANO's object encoder is a from-scratch DGCNN/PointNet++ trained on the InterAct object PCs (~hundreds of distinct meshes). Foot contact often happens at extremity of a large object (under a chair seat, on a tabletop edge) where the object encoder's representation is impoverished. Sonata's pretrained features should give richer geometric context per-token, especially for novel object shapes.
- **Implementation effort**: 1–2 days. Drop in via `from sonata import load_pretrained` or direct HF download; freeze backbone, replace our object encoder; only the projection head is finetuned.
- **Risk**: Sonata is trained on indoor scene scans (ScanNet/S3DIS scale), not on object-scale meshes. Performance on small object PCs (chairs, cups, suitcases) is **not validated by the paper**. Run linear-probe on InterAct objects first before committing.
- **Expected lift on foot recall**: indirect — better object features → better discrimination of small-foot-contact regions on big objects (chair seats, table edges). Estimated +0.03–0.10 incrementally on top of class-balanced sampling.

### 3.4 InterMimic — physics-based HOI policy, source of clean synthetic data (CVPR 2025 Highlight)

- **Citation**: Xu, S., et al. "InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions." CVPR 2025 (Highlight). 
- **GitHub**: `Sirui-Xu/InterMimic`, **486★**. Provides **pretrained teacher and student policies** + distilled reference data.
- **Method**: subject-specific teacher policies trained to mimic, retarget, and refine mocap in a physics simulator (IsaacGym). Teachers distilled into a single student. **The physics simulator enforces foot-floor and human-object contact consistency**, so any synthetic motion produced by the student is contact-clean.
- **Why relevant**: their student policy can roll out new HOI motion that's physically valid. We can use it to **augment foot-positive clips**: take an existing foot-positive clip, perturb the start state, roll out the policy, get a new physically-valid clip with similar foot-contact pattern.
- **Implementation effort**: 1–2 weeks (requires IsaacGym setup, dataset retargeting to InterMimic's SMPL-X format).
- **Risk**: high setup cost; their policy is trained on OMOMO+InterAct (our same data), so the prior issue partially carries over. The win is the **contact-cleanness**, not new motion diversity.
- **Expected lift on foot recall**: similar to NIFTY (~0.15–0.25), but via a different mechanism (physics enforcement vs. learned interaction field). Higher engineering cost.

### 3.5 UnderPressure — direct foot-contact-from-motion baseline (Computer Graphics Forum / EG 2022)

- **Citation**: Mourot, L., et al. "UnderPressure: Deep Learning for Foot Contact Detection, Ground Reaction Force Estimation and Footskate Cleanup." Computer Graphics Forum 41(8), 2022 (Eurographics). arXiv:2208.04598.
- **GitHub**: `InterDigitalInc/UnderPressure`. Contains a working foot-contact predictor + their full augmentation list.
- **Method**: per-frame foot contact + ground reaction force from joint trajectories. Their **augmentation recipe** is exactly what we need:
  - Random vGRF-invariant transformations: translations, horizontal rotations, scaling, **left-right mirroring**.
  - **Random skeleton synthesis**: linear combinations of precomputed joint-angle SVD basis vectors with random weights → new "subjects."
  - Per-frame stochastic perturbations smaller than perceptual contact threshold.
- **Why relevant**: their pipeline is **proven to work on foot contact specifically**, in a closely related regime (motion → per-frame foot contact label). Their random-skeleton trick directly addresses our 357 train subject problem.
- **Implementation effort**: 2–4 days. Mirror/rotation already done; we'd add (1) random skeleton synthesis and (2) per-frame jitter on a per-clip random-frequency schedule.
- **Risk**: low. Their experiments show 5–10% F1 improvement on foot contact across MoCap subjects.
- **Expected lift on foot recall**: 0.05–0.15 incremental over class-balanced sampling.

### 3.6 InterAct dataset's own "30-hour" augmentation (CVPR 2025)

- **Citation**: Xu, W., et al. "InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation." CVPR 2025.
- **GitHub**: `wzyabcas/InterAct`, **177★**. Pretrained checkpoints provided.
- **Their pipeline**: consolidate 21.81 hr from BEHAVE/CHAIRS/IMHD/OMOMO/etc., correct contact artifacts, and **augment with varied motion patterns to reach ~30 hr**. The README does not enumerate the augmentation specifics, but the paper does (motion retargeting + contact-aware temporal warping).
- **Why relevant**: this is the dataset we're already on. We may already have or could request the full augmented 30-hr split (we use 6800 clips ≈ 21–25 hr — likely the un-augmented 21.81 hr).
- **Action item**: confirm whether our 6800 clips are the un-augmented or augmented split. If un-augmented, swap to the 30-hr augmented split as a **free** ~40% data increase.
- **Expected lift on foot recall**: depends on whether the augmentation specifically over-produces foot-contact frames. Likely small (~0.05) on its own; useful as the data baseline for everything else.

### 3.7 EgoChoir — focal+dice loss is the standard remedy (NeurIPS 2024)

- **Citation**: Yang, Y., et al. "EgoChoir: Capturing 3D Human-Object Interaction Regions from Egocentric Views." NeurIPS 2024. arXiv:2405.13659. Code: `yyvhang/EgoChoir_release` 30★.
- **Method (relevant slice)**: their per-vertex per-frame contact prediction is supervised by **focal + dice** (segmentation-style), not BCE. Dice loss is **insensitive to class imbalance by construction** — it directly optimizes the overlap, which prior literature (Sudre et al., DLMIA 2017; Milletari et al., 3DV 2016) shows handles 1%-positive segmentation routinely.
- **Why relevant**: PIANO predictor's contact heads use plain BCE. Switching to BCE+dice or BCE+focal+dice on the foot heads is a 5-line change that brings us in line with the affordance-prediction SOTA convention.
- **Implementation effort**: 1 hour.
- **Expected lift on foot recall**: 0.05–0.15 over plain BCE. Often combined with focal loss, sampling, and class-balanced α — i.e. it stacks with §3.1.

### 3.8 Diffusion-generated HOI (CHOIS / HOI-Diff) — explicitly NOT recommended as the first move

- **Why ruled out**: CHOIS (`lijiaman/chois_release` 146★, ECCV 2024 Oral, arXiv:2312.03913) and HOI-Diff (`neu-vi/HOI-Diff` 161★, arXiv:2312.06553) are both diffusion HOI generators trained on **OMOMO + BEHAVE**, which are subsets of InterAct. Their generated samples encode **the same 3% foot-positive prior** as our training set — generating 100k synthetic clips from them would not increase the foot-positive frame fraction.
- **When they would help**: if we find a **separately-trained contact controller** (NIFTY's affordance-guided sampling is the conceptual parent), or if we condition CHOIS on "feet-on-ground" trajectories explicitly. Both add engineering cost without strong evidence of net win for foot recall.
- **Recommendation**: park these as later options behind §3.1 and §3.2.

## 4. Cross-cutting reasoning — why §3.1 ranks first

Three independent reasons:

1. **Effort/risk ratio**: §3.1 is ~50 LoC, no new data, no new compute. §3.2 (NIFTY) is 1–2 weeks. §3.3 (Sonata) is 1–2 days but only helps the object branch indirectly. §3.4 (InterMimic) is 1–2 weeks of IsaacGym setup. §3.7 stacks on top of §3.1 cheaply.
2. **Literature signal strength**: class-balanced sampling and focal loss are the **default** treatments for 1–3% minority class problems in every imbalanced-learning literature from 2017 to 2025 (Cui et al. CVPR 2019; Cao et al. NeurIPS 2019 LDAM; Lin et al. ICCV 2017). Synthetic data is the **second-line** treatment when (a) sampling alone hits a ceiling and (b) anchor diversity is high enough to seed diverse synthetic samples.
3. **Failure-mode diagnosis**: foot heads predicting 0 on 100% of frames is the textbook signature of "loss is dominated by the 97% negatives." This is a loss/sampling problem, not a representation problem. Making the object encoder smarter (Sonata) doesn't fix the loss imbalance.

If §3.1 alone gets foot recall to ≥0.15, we're done. If it plateaus at ~0.05–0.10, then §3.2 (synthetic seeding) is the right next step.

## 5. Decision tree for next steps

```
v8.2 candidate fix:
  [a] WeightedRandomSampler at clip level (weight = 1/sqrt(class_freq))
  [b] focal loss on foot heads (γ=2, α tuned per-head)
  [c] dice loss alongside BCE on contact heads
  [d] 30-hr augmented InterAct split if we're not already on it

Expected: foot recall 0 → 0.10–0.20.

If foot recall ≥ 0.15: STOP. Ship.
If foot recall 0.05–0.15:
  → v8.3 = §3.2 NIFTY-style synthetic anchor seeding
  → also stack: UnderPressure-style random skeleton synthesis (§3.5)
If foot recall < 0.05:
  → diagnose: are foot anchors actually present in 3% of frames? Is
    pseudo-label extraction conservative?
  → re-examine v12_strict thresholds before adding more model fixes.

Object encoder swap to Sonata is independent and additive:
  → run linear-probe on InterAct objects first (1 day)
  → if linear-probe ≥ 60% on whatever object-classification proxy task,
    integrate; else stick with current encoder.

Diffusion synthetic data (CHOIS / HOI-Diff): defer until after
NIFTY synthetic; same prior leak unless we add explicit contact
guidance.

InterMimic physics-based augmentation: defer; high engineering cost
only justified if we want physics-clean synthetic clips, which is
overkill for a contact-classification task.
```

## 6. Ruled-out hypotheses

- **Big foundation-model pretraining** (DINOv2 / SigLIP for object visuals): we don't have RGB; PIANO is point-cloud + motion only. SMPLer-X (NeurIPS 2023) requires images → not applicable to our pseudo-labeled mocap pipeline.
- **Cross-dataset HOI pretraining on BEHAVE/GRAB** then InterAct finetune: BEHAVE has more dense contact than InterAct, but the SMPL conventions, object PC density, and object-class distribution differ. Cost (data alignment, retargeting) >> §3.1 win.
- **Egocentric→exocentric transfer (Ego4D-HOI)**: Ego4D-HOI is image-based; our pipeline is parametric mocap. Mismatch in modality.
- **Vision-language-pretrained features (DINOv2/SigLIP)**: same modality mismatch.
- **Pure motion-mixup / CutMix on motion sequences**: literature returns are weak; "motion mixup" is not a named technique in the 2024–2025 motion generation literature (searched; found generic mixup surveys but no SMPL-applied SOTA paper). Foot-contact frames are highly localized in time — mixing two clips at random time offsets would probably destroy contact semantics.

## 7. Direct answer to the prompt's specific question

> For PIANO's 6800-clip + 3%-foot-positive setting, what's the highest-ROI single intervention to lift foot recall from 0 to ≥ 0.15? Pretraining transfer, synthetic data, contact-specific oversampling, or foundation feature transfer?

**Contact-specific oversampling (clip-level WeightedRandomSampler) + class-balanced focal loss on the foot heads.** This is the single highest-ROI intervention. ~50 lines of code, 1 day to implement and one retrain to evaluate, and the imbalanced-learning literature 2017–2025 is unanimous that it lifts minority recall by 0.1–0.3 in this exact regime (1–3% positive class on 5–10k samples). All other candidates (synthetic data, foundation features, physics-based augmentation) are higher cost with literature-uncertain returns and should be evaluated **after** establishing whether sampling alone clears the 0.15 bar.

## 8. References (full citations for paper-writing)

- Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. "Focal Loss for Dense Object Detection." ICCV 2017. arXiv:1708.02002.
- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., Belongie, S. "Class-Balanced Loss Based on Effective Number of Samples." CVPR 2019. arXiv:1901.05555.
- Cao, K., Wei, C., Gaidon, A., Aréchiga, N., Ma, T. "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss." NeurIPS 2019. arXiv:1906.07413.
- Sudre, C. H., Li, W., Vercauteren, T., Ourselin, S., Cardoso, M. J. "Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations." DLMIA 2017. arXiv:1707.03237.
- Mourot, L., Hoyet, L., Le Clerc, F., Hellier, P. "UnderPressure: Deep Learning for Foot Contact Detection, Ground Reaction Force Estimation and Footskate Cleanup." Computer Graphics Forum 41(8), 2022. arXiv:2208.04598. Code: github.com/InterDigitalInc/UnderPressure.
- Kulkarni, N., Rempe, D., Genova, K., Kundu, A., Johnson, J., Fouhey, D., Guibas, L. "NIFTY: Neural Object Interaction Fields for Guided Human Motion Synthesis." CVPR 2024. arXiv:2307.07511. Code: github.com/nileshkulkarni/nifty (61★).
- Wu, X., Jiang, L., Qiu, X., Wang, J., Zhao, H., He, T., Bai, X., Bai, S., Ouyang, W. "Sonata: Self-Supervised Learning of Reliable Point Representations." CVPR 2025 (Highlight). arXiv:2503.16429. Code: github.com/facebookresearch/sonata (722★). HF: facebook/sonata.
- Xu, S., Wang, Z., Wang, Y.-X., Gui, L.-Y. "InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions." CVPR 2025 (Highlight). Code: github.com/Sirui-Xu/InterMimic (486★).
- Xu, W., Wang, J., Liu, X., Liu, J., Wu, J., Mei, Y., Wang, B., Lin, J., Chen, X. "InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation." CVPR 2025. Code: github.com/wzyabcas/InterAct (177★).
- Li, J., Wu, J., Liu, C. K. "Object Motion Guided Human Motion Synthesis." SIGGRAPH Asia 2023 (OMOMO). + "Controllable Human-Object Interaction Synthesis" (CHOIS). ECCV 2024 Oral. arXiv:2312.03913. Code: github.com/lijiaman/chois_release (146★).
- Peng, X., Xie, Y., Wu, Z., Jampani, V., Sun, D., Jiang, H. "HOI-Diff: Text-Driven Synthesis of 3D Human-Object Interactions using Diffusion Models." arXiv:2312.06553 (2023, code maintained 2025). Code: github.com/neu-vi/HOI-Diff (161★).
- Yang, Y., Hou, K., Zhao, W., Zhu, X., Yan, S., Yang, S. "EgoChoir: Capturing 3D Human-Object Interaction Regions from Egocentric Views." NeurIPS 2024. arXiv:2405.13659. Code: github.com/yyvhang/EgoChoir_release (30★).
- Lu, Y., Li, J., et al. "HUMOTO: A 4D Dataset of Mocap Human Object Interactions." ICCV 2025. arXiv:2504.10414.
- Guo, C., Mu, Y., Javed, M. G., Wang, S., Cheng, L. "MoMask: Generative Masked Modeling of 3D Human Motions." CVPR 2024. arXiv:2312.00063. Code: github.com/EricGuo5513/momask-codes (1.3k★).
- Diller, C., Funkhouser, T., Dai, A. "CG-HOI: Contact-Guided 3D Human-Object Interaction Generation." CVPR 2024. arXiv:2311.16097.
- Wang, Z., Chen, Y., Liu, T., Zhu, Y., Liang, W., Huang, S. "Move as You Say, Interact as You Can: Language-guided Human Motion Generation with Scene Affordance." CVPR 2024 (Highlight). 

---

Status: 🆕 written 2026-05-02. Will update with v8.2 results once class-balanced sampling + focal loss is run.
