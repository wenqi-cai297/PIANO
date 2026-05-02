# 2026-05-02 — SOTA HOI affordance / contact prediction architectures (post Move-as-You-Say)

## TL;DR

Surveyed CVPR/ECCV/NeurIPS 2024 + CVPR/NeurIPS 2025 + 2025 arXiv for HOI
affordance/contact prediction architectures usable as a reference for
the v8 PIANO predictor (per-frame contact target on 128 object tokens).

**Direct answer to the gap question** ("who solves per-frame moving
contact?"): **EgoChoir (NeurIPS 2024)** is the closest published
analog — it predicts T-frame contact at SMPL-X vertex resolution +
per-vertex object affordance via parallel cross-attention. **GenHOI
(arXiv 2025-06)** is the next-closest, with per-frame Contact-Aware
HOI Attention. Most other SOTA methods (CG-HOI, CHOIS, HOI-Diff,
Text2HOI) collapse the temporal dimension by either max-pooling a
single contact map or predicting a small set of contact joints/points.

**v8's design (KL on 128 object tokens, per-frame, per-part) is
defensibly novel along one axis** — token-level instead of vertex-level
output. The literature does not show a consensus on whether this is
better; most SOTA prefers per-vertex heatmaps (Move-as-You-Say,
EgoChoir, Text2HOI), with v8's choice motivated by FPS-128 already
giving us a ~5cm spatial resolution on object-scale meshes.

## 1. Context

Symptom (`analyses/2026-05-05_predictor_v8_design.md`): per-part query
cross-attends to 128 FPS object tokens, KL-supervised against
σ=0.08m Gaussian-kerneled GT. Top-1 token recall = 9 % vs ideal 30-50 %;
top-3 = 21 %. Spatial concept exists but argmax is wrong.

Question: what do SOTA HOI affordance/contact predictors do, and is the
9 %→target gap achievable by switching architecture, supervision, or
output representation?

## 2. Strong candidates — full table

Filter: top venues 2024-2026, public GitHub (or strong arXiv with code
promised), affordance/contact prediction with explicit architecture and
loss. Ruthlessly cut: papers that only do final-frame contact, papers
that use only contact annotations as conditioning input not output,
blog/Medium content.

| Paper | Venue | Stars | Repr. | Per-frame? | Loss | Direct relevance |
|---|---|---:|---|---|---|---|
| **EgoChoir** | NeurIPS 2024 | 30 | Per-vertex on 6890 SMPL-X verts × T frames + per-point affordance on N obj points | ✓ T-frame | Focal + Dice (segmentation-style) | ★★★★★ |
| **Text2HOI** | CVPR 2024 | 117 | Per-point on object PC (N×1 sigmoid) via VAE prior | ✗ single-clip aggregate | BCE + Dice + KL (latent prior) | ★★★★ |
| **HOI-Diff (APDM)** | arXiv 2023, code maintained 2025 | 161 | 8 contact joints (binary) + 8 contact-point xyz (R³) | partial: per-clip but joint-level | MSE on joint contact + xyz, ε-prediction | ★★★ |
| **CHOIS** | ECCV 2024 (Oral) | 146 | BPS (Basis Point Set) per object — implicit contact via geometry | ✗ no explicit contact head | object-geometry loss + waypoint loss | ★★ (different paradigm) |
| **GenHOI / ContactDM** | arXiv 2025-06 | code coming | Concatenated human+obj PC features per-frame, attended by pose query | ✓ per-frame | diffusion ε-pred + reconstruction guidance | ★★★★ |
| **CG-HOI** | CVPR 2024 | n/a (no code release) | Body-vertex + object-vertex contact, joint diffusion | ✓ per-frame (joint diffusion) | diffusion + cross-attn between human/obj/contact branches | ★★★ |
| **Move as You Say** | CVPR 2024 (Highlight) | 176 | Per-scene-point distance heatmap (ADM = Affordance Diffusion Model) | ✗ scene-level single map | diffusion ε-pred on heatmap | ★★★ (anchor only) |

## 3. Per-paper deep dive (top 5)

### 3.1 EgoChoir — NeurIPS 2024 [strongest match for our problem]

- **Citation**: Yang, Y., Hou, K., Zhao, W., Zhu, X., Yan, S., Yang, S.
  "EgoChoir: Capturing 3D Human-Object Interaction Regions from
  Egocentric Views." NeurIPS 2024. arXiv 2405.13659.
- **GitHub**: `yyvhang/EgoChoir_release`, **30 stars** (low because
  niche egocentric setup, but code is real and complete).
- **Output shape**:
  - Object affordance: φ_a ∈ R^(N×1) — per-point on N PC points
    (DGCNN-encoded), single sigmoid per point (interaction probability).
  - Human contact: φ_c ∈ R^(T × 6890 × 1) — **temporally dense per-vertex
    SMPL-X contact across T frames**.
- **Architecture**: parallel cross-attention. Semantic token T_f
  concatenated with object features F_O is the query; visual features
  F_V and motion features F_M are two parallel KV pairs. **Modulation
  tokens (τ_v, τ_m) gradient-modulate which clue dominates**, allowing
  the network to fall back to motion when visual is occluded.
- **Loss**: focal + dice (segmentation-style) on both contact and
  affordance, plus KL on motion-encoder discrepancy, plus CE on the
  semantic head.
- **Relevance to v8**:
  - **Confirms per-frame contact + per-point object affordance is the
    right factorization** — they predict both jointly.
  - Their "modulation token" trick directly answers **how to handle
    moving contact** that v8 currently struggles with: they explicitly
    bias which signal (visual vs motion) is trusted per-frame. PIANO
    has analogous text + initial-pose split that could be modulated.
  - **Their loss choice diverges from ours** — focal+dice instead of
    KL. Focal pushes the model harder on the hard examples (the
    spatially-precise contact frames) and dice is robust to class
    imbalance (most points are non-contact). Both are likely reasons
    they get sharper attention than our soft-KL.
  - **Output is per-vertex, not per-token**: 6890 verts with
    geodesic-distance neighborhood. We're at 128 tokens (~50× sparser).
    This is a degree-of-freedom they spend that we don't.

### 3.2 Text2HOI — CVPR 2024

- **Citation**: Cha, J., Kim, J., Yoon, J.S., Baek, S. "Text2HOI:
  Text-guided 3D Motion Generation for Hand-Object Interaction." CVPR
  2024. arXiv 2404.00562.
- **GitHub**: `JunukCha/Text2HOI`, **117 stars**.
- **Output shape**: m_contact ∈ R^(N×1), per-point on FPS-N object PC.
  Single sigmoid per point.
- **Crucially**: this is a **single-clip aggregate** — represents
  "regions touched at any time during interaction". They get away with
  this because hand-object interaction has a small number of distinct
  contact regions per clip; for whole-body PIANO with sit + grasp
  phases this would underspecify the target.
- **Architecture**: VAE — input is object PC + text + scale, with a
  64-d Gaussian latent z_contact sampled from a learned prior.
  Decoder is a MLP over PC features.
- **Loss**: BCE + dice + KL (KL is on the latent, not the contact map
  itself).
- **Relevance to v8**:
  - **Validates BCE + dice for sparse contact maps**, which our
    σ=0.08m KL setup avoids. Worth piloting.
  - **VAE latent is overkill for our setting** — they need it for
    diversity in hand grasp synthesis; we have one ground-truth
    contact target per (frame, part).
  - **Their per-point output dimension (N) is similar to our 128
    tokens**, so the resolution trade-off is comparable. They report
    sharper contact than typical regression — implies our token
    granularity is fine, the supervision is the issue.

### 3.3 HOI-Diff (APDM) — arXiv 2023 (still maintained 2025)

- **Citation**: Peng, X., Xie, Y., Wu, Z., Jampani, V., Sun, D.,
  Jiang, H. "HOI-Diff: Text-Driven Synthesis of 3D Human-Object
  Interactions using Diffusion Models." arXiv 2312.06553.
- **GitHub**: `neu-vi/HOI-Diff`, **161 stars**, README says CVPR 2025
  HuMoGen workshop.
- **Output shape**: 8 contact-joint binary labels y_h ∈ {0,1}^8 + 8
  contact-point xyz on object surface y_o ∈ R^(8×3) + binary
  static/dynamic. **Sparsely placed**: 8 joints × 1 contact-point each.
- **Loss**: MSE on the diffusion ε-prediction over the joint+xyz tuple.
- **Relevance to v8**:
  - **They downcast contact prediction to 8 sparse joint × xyz pairs**,
    which sidesteps the 128-token-distribution problem entirely.
  - Could be a **fallback if v8 KL-on-128-tokens doesn't lift** — turn
    target into "per-part xyz regression on object surface", but with
    an APDM-style diffusion head instead of a single linear projection.
  - **They threshold at τ=0.6 to gate which joints are active** — this
    is the same contact-gate pattern v7-fix uses, validates it is a
    real pattern in the field.
  - **Diffusion architecture is heavy** — 8 transformer layers + 500
    diffusion steps for a head that's smaller than our predictor's.
    Probably overkill for our setting, but the per-frame contact
    architecture deserves an ablation.

### 3.4 GenHOI — arXiv 2025-06 [latest]

- **Citation**: Liu, Y. (and others), "GenHOI: Generalizing
  Text-driven 4D Human-Object Interaction Synthesis for Unseen
  Objects." arXiv 2506.15483 (June 2025).
- **GitHub**: project page exists, code not yet released as of fetch
  date. Worth checking again in 6 months.
- **Architecture**: Two-stage. Stage 1 = **Object-AnchorNet** outputs
  K=5 sparse 3D HOI keyframes from text + object. Stage 2 =
  **ContactDM** = Contact-Aware Diffusion Model that interpolates 5
  keyframes → 120-frame outputs. Inside ContactDM:
  - **Contact-Aware Encoder** takes concatenated human+object PC with
    one-hot indicator → per-frame feature F_i ∈ R^d via PointNet++.
  - **Contact-Aware HOI Attention**: pose embedding E_pose is the
    query; F_i is K and V. Cross-attention produces motion that
    "attends to critical contact regions".
- **Per-frame**: yes, F_i is per-frame.
- **Loss**: standard diffusion ε-prediction; no separate explicit
  contact loss reported in the paper.
- **Relevance to v8**:
  - **Their factorization (sparse keyframes → dense interpolation) is
    a different decomposition** than ours but might apply: predict
    contact at 5-10 anchor frames then interpolate. Worth holding as
    a v9+ alternative if v8's per-frame loss is too noisy.
  - **They concatenate human + object PC into a single feature
    stream**, instead of cross-attending. Different choice from ours,
    but they're solving a different problem (motion interpolation,
    not contact target).

### 3.5 CG-HOI — CVPR 2024

- **Citation**: Diller, C., Dai, A. "CG-HOI: Contact-Guided 3D
  Human-Object Interaction Generation." CVPR 2024. arXiv 2311.16097.
- **GitHub**: **No code release** (project page only —
  cg-hoi.christian-diller.de). Drops the candidate to ★★★ despite
  strong design.
- **Output**: contact predicted on both human-mesh vertices and
  object-mesh vertices, jointly with motion in a unified diffusion.
- **Cross-attention** between three branches: human motion, object
  motion, contact.
- **Relevance to v8**:
  - **Architecture validates separate-but-cross-attended contact
    branch**, which is what our StructuredHead does.
  - No code → not actionable as a reference implementation.

## 4. Other papers checked + ruled out

| Paper | Venue | Reason ruled out |
|---|---|---|
| AffordanceLLM | CVPR 2024 (workshop) | 2D affordance from VLM, single-image — not 3D nor temporal |
| SceneVerse | ECCV 2024 | scene grounding dataset, not affordance prediction |
| HOIAnimator | CVPR 2024 | dual diffusion for human + object motion, contact handled by "perceptive message passing" — no separate contact head, just consistency loss |
| InterDiff | ICCV 2023 | older than CVPR 2024 cutoff; physics correction not contact prediction |
| InterDreamer | NeurIPS 2024 | zero-shot LLM planning + retrieval; no learned contact predictor |
| InterMimic | CVPR 2025 (Highlight) | physics-based RL controller, not a contact predictor |
| InterAct | CVPR 2025 | benchmark + dataset paper; multi-task baselines but no novel contact predictor |
| OnlineHOI | arXiv 2509 | Mamba-based online setting, contact handled implicitly through memory |
| THOR | arXiv 2403 | text-to-HOI diffusion with relation intervention, but no explicit contact predictor |
| ParaHome | NeurIPS 2024 | dataset paper |
| OakInk2 | CVPR 2024 | dataset + TaMF baseline, no novel contact head |
| TokenHSI | CVPR 2025 | RL multi-skill humanoid policy, not contact prediction |
| HUMOTO | arXiv 2504 | dataset paper |
| Populate-A-Scene | arXiv 2507 | video synthesis, not contact prediction |
| DAViD | arXiv 2501 | dynamic affordance via 2D video diffusion lift to 3D — interesting but not direct match |
| Vision-Guided Action / GAP3DS | CVPR 2025 | gaze-informed motion prediction, single human, no per-frame object contact |
| AffordDP | CVPR 2025 | robotic manipulation policy, not body-mesh contact |
| LMAffordance3D | CVPR 2025 | 3D affordance grounding from language, single-shot, no temporal |

## 5. Cross-cutting observations

### 5.1 Output representation consensus

| representation | papers | strength | weakness |
|---|---|---|---|
| Per-vertex heatmap on object/body mesh | EgoChoir, Move-as-You-Say, CG-HOI | Highest spatial resolution, geodesic-aware | Costly when N=6890; heavy decoder |
| Per-token soft attention | **PIANO v8** (only) | Cheap (128 tokens), uses transformer naturally | Coarse spatial resolution (~5cm), no geodesic awareness |
| Per-FPS-point on object PC | Text2HOI, EgoChoir (object side) | Mid-cost (N≈1024-4096) | Loses fine geometry; FPS coverage inconsistent |
| Sparse contact joints + xyz | HOI-Diff (8 joints × 3 xyz) | Trivial to predict, easy to gate | Loses spatial context entirely |

**Verdict**: per-token attention (v8's choice) is rare in the
literature. Most SOTA goes per-vertex or per-point at higher
resolution. Whether v8's coarser 128-token output is the bottleneck
or the supervision is the bottleneck is a real open question.

### 5.2 Loss function consensus

| loss | papers | rationale |
|---|---|---|
| **Focal + dice** | EgoChoir | imbalanced binary contact, hard-example focus |
| **BCE + dice + KL** | Text2HOI | combines pointwise + region-wise + diversity |
| **MSE / ε-prediction** | HOI-Diff, CG-HOI, GenHOI, Move-as-You-Say | when contact is part of a diffusion framework, the diffusion loss subsumes contact loss |
| **KL divergence (Gaussian-kernelled GT)** | **PIANO v8** | analogous to soft-label distillation |

**Verdict**: KL-divergence on a Gaussian-kernelled GT is **not the
mainstream choice**. The mainstream is focal+dice (when binary
contact) or MSE-via-diffusion (when contact is generated). v8's KL
choice is defensible (it's natural for soft attention + softmax) but
worth ablating against focal+dice.

### 5.3 Per-frame moving contact — the gap

Only EgoChoir and CG-HOI predict T-frame per-vertex contact.

- **EgoChoir's solution**: parallel cross-attention with
  modulation-gated KV streams. Visual + motion clues parallel-attend
  the object features, with τ-modulated gradient.
- **CG-HOI's solution**: joint diffusion of human, object, and
  contact, with cross-attention between branches. Per-frame contact
  emerges from the diffusion process, not from a separate predictor.
- **PIANO v8's solution**: cross-attention from per-frame query token
  to 128 object tokens, KL-supervised with Gaussian kernel.

The PIANO v8 solution is **architecturally lightest** of the three.
The 9 % top-1 recall suggests this is too light. Two upgrade paths
inspired by the SOTA:

1. **EgoChoir-style**: add a parallel motion stream (initial pose +
   text encoder) with τ-gated KV fusion. Cost: +1 cross-attention
   block per part. Likely lift on hand/foot which need motion clue.
2. **HOI-Diff-style**: replace soft-KL with a small diffusion head
   over the 128-token logit. Cost: 8 transformer layers + 50 sampling
   steps at inference. Likely lift on argmax sharpness.

### 5.4 Why pelvis works but hand/foot doesn't (PIANO observation)

Cross-referencing the paper observations: pelvis-contact (sit) is
**spatially stationary across many frames** of stable_contact phase.
EgoChoir explicitly notes that head-motion modulation helps when the
visual signal is ambiguous — exactly when motion is happening. Hand/
foot in PIANO move during contact (lifting, pushing); without a
motion-fused KV stream the model has no way to track the moving
target across frames. **This is consistent with the pattern in
EgoChoir** and is likely the architectural reason for our per-part
gap.

## 6. Recommended next steps for v8/v9

**Without changing v8 trajectory** — order by impact × effort:

1. **Add focal-loss option** to KL on contact-positive cells. EgoChoir
   evidence suggests this sharpens argmax. Cost: ~10 lines in
   `losses.py`. Run as A/B during v8 training.
2. **Add motion-stream cross-attention** (EgoChoir-style modulation):
   per-frame text embedding + per-frame initial-pose encoding fed as
   second KV in addition to object tokens. Per-part query attends to
   both. Cost: +1 cross-attn block. Defer to v9 unless v8 misses
   gate.
3. **Higher token resolution** (FPS 256 or 512) if v8 hits gate but
   target_top1 still <30 %. EgoChoir uses 6890 verts; even 4× more
   than 128 would help.
4. **Switch target loss to focal-dice** if KL underperforms. Validated
   in both EgoChoir + Text2HOI.

**Defer to v10+**:
- Diffusion head over contact tokens (HOI-Diff-style). Heavy.
- Sparse-keyframe + interpolation (GenHOI-style). Architectural shift.

## 7. References (full citations)

### Primary candidates

1. Yang, Y., Hou, K., Zhao, W., Zhu, X., Yan, S., Yang, S. **"EgoChoir:
   Capturing 3D Human-Object Interaction Regions from Egocentric
   Views."** NeurIPS 2024. arXiv: 2405.13659.
   Code: github.com/yyvhang/EgoChoir_release (30★).
2. Cha, J., Kim, J., Yoon, J.S., Baek, S. **"Text2HOI: Text-guided 3D
   Motion Generation for Hand-Object Interaction."** CVPR 2024.
   arXiv: 2404.00562. Code: github.com/JunukCha/Text2HOI (117★).
3. Peng, X., Xie, Y., Wu, Z., Jampani, V., Sun, D., Jiang, H.
   **"HOI-Diff: Text-Driven Synthesis of 3D Human-Object Interactions
   using Diffusion Models."** CVPR 2025 HuMoGen Workshop. arXiv:
   2312.06553. Code: github.com/neu-vi/HOI-Diff (161★).
4. Li, J., Clegg, A., Mottaghi, R., Wu, J., Puig, X., Liu, K.
   **"Controllable Human-Object Interaction Synthesis."** ECCV 2024
   (Oral). arXiv: 2312.03913. Code: github.com/lijiaman/chois_release
   (146★).
5. Liu, Y. (et al.) **"GenHOI: Generalizing Text-driven 4D
   Human-Object Interaction Synthesis for Unseen Objects."** arXiv:
   2506.15483 (June 2025). Code: pending.
6. Diller, C., Dai, A. **"CG-HOI: Contact-Guided 3D Human-Object
   Interaction Generation."** CVPR 2024. arXiv: 2311.16097. No code.
7. Wang, Z., Chen, Y., Jia, B., Li, P., et al. **"Move as You Say,
   Interact as You Can: Language-guided Human Motion Generation with
   Scene Affordance."** CVPR 2024 (Highlight). arXiv: 2403.18036.
   Code: github.com/afford-motion/afford-motion (176★).

### Ruled-out / context-only

8. Xu, S., Wang, Z., Wang, Y.-X., Gui, L.-Y. **"InterDreamer:
   Zero-Shot Text to 3D Dynamic Human-Object Interaction."** NeurIPS
   2024. arXiv: 2403.19652.
9. Xu, S., et al. **"InterMimic: Towards Universal Whole-Body Control
   for Physics-Based Human-Object Interactions."** CVPR 2025
   (Highlight). arXiv: 2502.20390. Code:
   github.com/Sirui-Xu/InterMimic.
10. Xu, W., et al. **"InterAct: Advancing Large-Scale Versatile 3D
    Human-Object Interaction Generation."** CVPR 2025. arXiv:
    (CVPR proceedings). Code: github.com/wzyabcas/InterAct.
11. Qian, S., Chen, W., Bai, M., Zhou, X., Tu, Z., Li, L.E.
    **"AffordanceLLM: Grounding Affordance from Vision Language
    Models."** CVPR 2024 OpenSUN3D Workshop. arXiv: 2401.06341. Code:
    github.com/JasonQSY/AffordanceLLM.
12. Jia, B., et al. **"SceneVerse: Scaling 3D Vision-Language Learning
    for Grounded Scene Understanding."** ECCV 2024. arXiv: 2401.09340.
13. Song, W., et al. **"HOIAnimator: Generating Text-Prompt
    Human-Object Animations Using Novel Perceptive Diffusion Models."**
    CVPR 2024.
14. Zhu, H., et al. **"LMAffordance3D: Grounding 3D Object Affordance
    with Language Instructions, Visual Observations and
    Interactions."** CVPR 2025.
15. Wu, Y., et al. **"Vision-Guided Action: Enhancing 3D Human Motion
    Prediction with Gaze-informed Affordance in 3D Scenes."** CVPR 2025.
16. Wu, X., et al. **"AffordDP: Generalizable Diffusion Policy with
    Transferable Affordance."** CVPR 2025.

### Architectural anchors (referenced for design lineage)

17. Xu, D., Ouyang, W., Wang, X., Sebe, N. **"PAD-Net: Multi-tasks
    Guided Prediction-and-Distillation Network for Simultaneous Depth
    Estimation and Scene Parsing."** CVPR 2018.
18. Bengio, S., Vinyals, O., Jaitly, N., Shazeer, N. **"Scheduled
    Sampling for Sequence Prediction with Recurrent Neural Networks."**
    NeurIPS 2015. arXiv: 1506.03099.

## 8. Open questions for follow-up

1. Did EgoChoir specifically ablate focal vs KL? If yes, what was the
   gap? — Need to read the full PDF.
2. CG-HOI's no-code policy means we can't easily reproduce; is there
   an equivalent open-source fork?
3. GenHOI's keyframe + interpolation paradigm — is the keyframe count
   K=5 a hyperparameter or specific to their dataset? Could PIANO use
   K=10?
4. For the per-frame moving contact gap (hand/foot weak in PIANO):
   does adding a motion stream (initial pose + text per-frame) close
   it, or is the issue in the trunk transformer's 196-frame attention
   span?

