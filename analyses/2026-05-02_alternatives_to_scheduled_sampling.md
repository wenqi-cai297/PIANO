# 2026-05-02 — SOTA alternatives to scheduled sampling for v8 StructuredHead

## TL;DR

The v8 design (`analyses/2026-05-05_predictor_v8_design.md` §3.4) currently
uses **Bengio NeurIPS 2015 scheduled sampling** to anneal teacher-forcing
on the contact -> {target, phase} -> support DAG. This is now considered
dated — 2024-2026 has produced four families of alternatives that
explicitly target the train-test gap:

1. **Mask-predict / iterative parallel decoding** (MaskGIT lineage)
2. **Discrete-state / masked diffusion** (SEDD, MDLM, LLaDA, BD3LMs)
3. **Self-rollout training** ("self-forcing" family — Diffusion Forcing,
   Self-Forcing) — explicitly framed as "bridging train-test gap"
4. **Multi-task denoising diffusion** (DiffusionMTL, TaskDiffusion) —
   structured prediction-specific, supports partial labels

For the v8 setup (T=196 frames, 5 contact parts × 3 phases × 4 support
states + 128-token target attention, sparse 50 % contact pseudo-labels,
explicit DAG), the strongest fit is a **frame-level mask-predict /
masked-diffusion training over the (contact, phase, support) joint
label tensor**, with **target attention conditioned on the unmasked
joint state**. This replaces "feed GT contact" with "feed *partially
masked* GT contact + phase + support, predict everything jointly". See
§4 for fit analysis and §6 for the recommended v9 design sketch.

## 1. The problem with Bengio 2015 scheduled sampling

### 1.1 Theoretical issue

Huszár "How (not) to Train your Generative Model" (arXiv:1511.05101, 2015)
proved scheduled sampling is **not a consistent estimator** of the
underlying conditional distribution. The mixed sampling distribution it
trains on does not converge to the true data distribution as
teacher-forcing-prob → 0. This is not a "tuning" issue, it is structural.

### 1.2 Empirical issue in our v8 setup

The schedule (epochs 0-50 TF=1.0, 50-80 anneal to 0.5, 80-100 hold) means:

- For 50 epochs, downstream heads (target, phase, support) only ever see
  *clean* upstream signals. They never learn to recover from a wrong
  contact prediction.
- After anneal, when contact head is wrong (frequent — `contact macro_f1 =
  0.237` in v7-fix), downstream heads see a distribution they were never
  trained on.

This shows up as: even when contact head improves marginally, target /
phase / support stay flat or regress vs the naive parallel head, because
their adaptation was capped at TF=0.5 noise level.

### 1.3 What modern alternatives change

All four families below share one principle: **the downstream head sees
the same distribution at train and test time, by construction**. Either
(a) both train and test see partially-corrupted inputs (mask-predict /
masked diffusion), (b) both see model rollouts (self-forcing), or
(c) the dependency is reformulated as joint denoising rather than
sequential conditioning (TaskDiffusion).

## 2. Curated candidate methods (ranked by fit to v8)

### 2.1 [Rank 1] MoMask: Generative Masked Modeling of 3D Human Motions

- **Citation**: Guo, C., Mu, Y., Javed, M.G., Wang, S., Cheng, L.
  "MoMask: Generative Masked Modeling of 3D Human Motions." CVPR 2024.
  arXiv: 2312.00063.
- **Code**: [EricGuo5513/momask-codes](https://github.com/EricGuo5513/momask-codes)
  — **1.3k stars**, 106 forks, active.
- **Innovation** (2-3 sentences): Hierarchical residual VQ-VAE
  tokenizes motion into discrete codes; a bidirectional transformer is
  trained with a *random* mask ratio per batch (uniform in [0,1])
  and predicts masked tokens conditioned on text and unmasked tokens.
  Inference iteratively unmask-predicts (cosine mask schedule, ~10
  steps), starting from fully masked. **No teacher forcing
  anywhere** — the model is always doing the same task at train and
  test: "given partial sequence, fill the holes."
- **Why it could replace scheduled sampling for v8**: The v8 DAG
  problem (contact + phase + support are causally linked and need
  consistent joint prediction over T=196 frames) maps directly to
  MoMask's setup (motion tokens are causally linked, predicted jointly).
  Replace "feed GT contact, predict target" with "feed a randomly
  masked subset of (contact, phase, support, target_token), predict
  the masked entries". The same trunk + structured head can be re-used.
- **Direct domain transfer**: MoMask works on T=196 HumanML3D motion
  sequences — exactly our T. The base codebase even includes the
  HumanML3D data loader our project may already use.
- **Caveat**: Requires discretizing the target into tokens (already
  done in v8 — 128 affordance tokens). Contact / phase / support are
  already categorical so this is free.

### 2.2 [Rank 2] LLaDA: Large Language Diffusion Models

- **Citation**: Nie, S., Zhu, F., et al. "Large Language Diffusion
  Models." ICLR 2025 (and presented at ICML 2025). arXiv: 2502.09992.
- **Code**: [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) —
  **3.8k stars**, 232 forks, very active community (LLaDA-V vision
  extension already exists).
- **Innovation**: 8B-scale masked diffusion LM trained from scratch
  matching LLaMA3 8B. The training objective is the *negative ELBO*
  for absorbing-state discrete diffusion: at each step, sample a mask
  rate t ∈ U[0,1], mask each token independently with prob t, predict
  all masked tokens with cross-entropy on those positions only.
  At inference, iteratively predict-then-remask (low-confidence tokens
  remasked first).
- **Why it could replace scheduled sampling**: The training objective
  has *no notion* of teacher forcing. The model sees a partial
  sequence, predicts the full joint distribution over masked entries.
  Cross-task DAG dependencies emerge implicitly from the joint
  prediction — when contact is masked but phase is given, the model
  learns P(contact | phase); when phase is masked and contact given,
  it learns P(phase | contact). The full DAG is captured by training
  on all mask patterns.
- **Fit for v8**: Strong but requires re-architecting the head as a
  joint token-prediction head over (contact, phase, support, target).
  Full v9 redesign rather than a v8 tweak. Bonus: would naturally
  handle the **sparse 50 % contact** regime — non-contact frames are
  just another "context" the model conditions on.

### 2.3 [Rank 3] Diffusion Forcing

- **Citation**: Chen, B., Monsó, D.M., Du, Y., Simchowitz, M.,
  Tedrake, R., Sitzmann, V. "Diffusion Forcing: Next-Token Prediction
  Meets Full-Sequence Diffusion." NeurIPS 2024. arXiv: 2407.01392.
- **Code**: [buoyancy99/diffusion-forcing](https://github.com/buoyancy99/diffusion-forcing)
  — **1.2k stars**, 69 forks, MIT CSAIL backed.
- **Innovation**: Trains a single model to denoise tokens at
  *independent per-token noise levels* over a sequence. This unifies
  next-token prediction (one token at noise=0, next at noise=1) and
  full-sequence diffusion (all tokens at the same noise level) as
  special cases of one training objective. At inference you can
  freely choose the schedule.
- **Why it could replace scheduled sampling**: The "tokens at
  different noise levels" framing maps almost perfectly to our DAG.
  Train with: contact at noise=t1, phase at noise=t2, support at
  noise=t3, target at noise=t4, sampled jointly each batch. The model
  must learn to be robust to *any* mixture of noisy and clean
  upstream signals — strictly stronger than scheduled sampling's
  binary on/off teacher forcing.
- **Fit for v8**: Excellent conceptual fit; moderate engineering
  cost. The 5-state contact / 3-state phase / 4-state support are
  small enough that a per-noise-level conditioning embedding is
  cheap. The MIT robotics applications (Franka swap task) already
  validate the framework on multi-modal structured outputs.
- **Caveat**: Repo is research-quality, not as battle-tested as
  MoMask for our exact task domain.

### 2.4 [Rank 4] DiffusionMTL: Multi-Task Denoising Diffusion from Partially Annotated Data

- **Citation**: Ye, H., Xu, D. "DiffusionMTL: Learning Multi-Task
  Denoising Diffusion Model from Partially Annotated Data." CVPR 2024.
  arXiv: 2403.15389.
- **Code**: [prismformore/DiffusionMTL](https://github.com/prismformore/DiffusionMTL)
  — moderate stars (verify directly), official.
- **Innovation**: Reformulates partial-label multi-task dense
  prediction as a *pixel-level joint denoising* problem. Each task
  prediction is a noisy version of the GT; a multi-task UNet learns
  to jointly denoise all tasks, with a shared backbone and
  task-specific decoders. Loss is computed only on annotated cells
  for each task.
- **Why it directly addresses our problem**: The v8 setup has
  **exactly** the partial-annotation pattern DiffusionMTL targets:
  some clips have contact labels but not target, some have target
  but unreliable phase, etc. Joint denoising naturally handles this.
  No teacher forcing because there is no autoregressive structure —
  all tasks are denoised jointly per pixel/frame.
- **Fit for v8**: Strongest *conceptual* match for the multi-task +
  partial label problem. Weaker fit for the *temporal* T=196 axis
  (DiffusionMTL is image-level, not sequence-level), so would need
  adaptation.
- **Caveat**: Smaller community than the diffusion-LM works, but
  paper is CVPR 2024 with reproducible eval on PASCAL-Context and
  NYUD-v2.

### 2.5 [Rank 5] Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion

- **Citation**: Huang, X., He, G., et al. "Self Forcing: Bridging the
  Train-Test Gap in Autoregressive Video Diffusion." NeurIPS 2025
  Spotlight. arXiv: 2506.08009.
- **Code**: [guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing)
  — **3.3k stars**, 263 forks, very active.
- **Innovation**: Conditions each frame's generation on **previously
  self-generated outputs** (KV-cached rollout) during training,
  rather than on GT context. Adds a holistic video-level loss instead
  of pure frame-wise objectives. Uses a stochastic gradient
  truncation to make this trainable.
- **Why it could replace scheduled sampling**: It *is* the
  generalization of "TF=0" — every step, the downstream head sees
  the upstream head's actual prediction. The video-level loss is the
  on-policy analog of REINFORCE / DAgger but differentiable. Avoids
  the consistency problem of scheduled sampling because there is no
  mixing.
- **Fit for v8**: Conceptually right but framework-heavy. Best fit
  for v10+ when v8/v9 evidence has accumulated. Most useful if we
  later need a Stage A predictor that *generates* sequences (rolling
  prediction), not just per-frame classifies.

### 2.6 [Rank 6] Score Entropy Discrete Diffusion (SEDD)

- **Citation**: Lou, A., Meng, C., Ermon, S. "Discrete Diffusion
  Modeling by Estimating the Ratios of the Data Distribution." ICML
  2024 **Best Paper**. arXiv: 2310.16834.
- **Code**: [louaaron/Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)
  — **725 stars**, foundational reference impl.
- **Innovation**: Defines "score entropy", a tractable score-matching
  objective for discrete data, replacing the looser ELBO-bound used
  by D3PM / MDLM / LLaDA. Becomes the theoretical foundation for the
  later masked-diffusion LMs. Beats GPT-2 with the same compute,
  reduces perplexity 25-75 % vs prior discrete diffusion.
- **Why it could replace scheduled sampling**: Same as LLaDA / MDLM —
  no autoregressive prefix, joint prediction at each step. SEDD adds
  a tighter loss that makes training more stable for small categorical
  outputs (5 / 3 / 4 classes — exactly our regime).
- **Fit for v8**: Use the SEDD score-entropy loss as the **target
  loss** for contact + phase + support joint distribution. Cleaner
  theoretical grounding than the v8 cross-entropy. Implementation
  cost is moderate; the loss formula is one function in `losses.py`.

### 2.7 [Rank 7] MDLM: Simple and Effective Masked Diffusion Language Models

- **Citation**: Sahoo, S.S., Arriola, M., Gokaslan, A., et al.
  "Simple and Effective Masked Diffusion Language Models." NeurIPS
  2024. arXiv: 2406.07524.
- **Code**: [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm)
  — **245-670 stars** (sources disagree, growing). Powers ByteDance
  Seed Diffusion + NVIDIA Genmol.
- **Innovation**: Rao-Blackwellized masked diffusion objective
  reduces to a *mixture of weighted MLM losses* — extremely simple to
  implement, much more stable than continuous-time discrete diffusion.
  Beats AR perplexity on Lambada / Scientific Papers.
- **Why it could replace scheduled sampling**: Same masked-diffusion
  theme but with the cleanest, simplest training recipe of the four
  (LLaDA / SEDD / BD3LMs / MDLM). One-line loss, no special
  schedule — just Bernoulli mask, predict, weight by 1/(1-α_t).
- **Fit for v8**: Best engineering cost / conceptual cleanliness
  trade-off if you want to prototype "joint mask-predict over
  (contact, phase, support, target)" without committing to a full
  diffusion framework. Direct drop-in replacement for cross-entropy
  on each head.

## 3. Honourable mentions (did not make top 7)

- **Block Diffusion (BD3LMs)** — Arriola et al., **ICLR 2025 Oral**,
  arXiv: 2503.09573, [kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms).
  Hybrid AR-block + diffusion-within-block. Useful if we later want
  *streaming* prediction over T=196. For v8 (parallel per-frame),
  pure masked diffusion (LLaDA/MDLM) is simpler.
- **TaskDiffusion** — Yang, Y., et al., NeurIPS 2024 (under review at
  the time), [YuqiYang213/TaskDiffusion](https://github.com/YuqiYang213/TaskDiffusion).
  Joint denoising for multi-task dense prediction. Fewer stars,
  newer; same family as DiffusionMTL with a different decoder.
- **MaskGIT** — Chang et al., CVPR 2022 (the seminal paper).
  Foundational reference for the iterative parallel decoding paradigm.
  All 2024-2025 motion / language masked-diffusion work derives from
  it. Cite for context but not directly the strongest replacement
  candidate (the descendants are stronger).
- **Elucidating Exposure Bias in Diffusion Models** — Ning et al.,
  ICLR 2024, arXiv: 2308.15321. Diagnostic + epsilon-scaling fix.
  Useful theoretical citation but not a structured-prediction method.
- **Self-Distillation Fine-Tuning (SDFT)** — Yang et al., 2024.
  Distribution-aware target alignment. Would only work as an
  *adjunct* to keep v8's scheduled-sampling cap (not a replacement
  primary method).

## 4. The specific question: multi-task structured prediction with explicit cross-task DAG and NO teacher forcing — does it exist (2024-2026)?

**Yes — three papers fit this exactly:**

1. **DiffusionMTL** (Ye & Xu, CVPR 2024). Multi-task partial-label
   dense prediction. Joint UNet denoising across tasks with shared
   backbone. **No teacher forcing** — every task is predicted jointly
   from a noisy state at every step. The cross-task dependency is
   modelled implicitly through the shared denoising pathway and
   per-task masks for partial labels.
   Training: uniform-time DDPM-style noise sampling on each task's
   prediction, masked by annotation availability.

2. **TaskDiffusion** (Yang et al., NeurIPS 2024 submission, OpenReview
   ID TzdTRC85SQ). Joint denoising over multi-task labels with a
   "task-integration" feature space; explicit cross-task interaction
   in the decoder.
   Training: same DDPM-style joint denoising, no autoregressive
   prefix.

3. **MoMask** (Guo et al., CVPR 2024). Although nominally
   single-task (motion-from-text), its hierarchical RVQ structure
   imposes **explicit DAG**: base-layer tokens condition residual
   layers. The masked transformer for the base layer trains with
   *random mask ratio*, never feeding GT residuals to the base.
   Training: P(masked tokens | unmasked tokens, text) cross-entropy
   on a uniform mask-ratio distribution.

The training mechanism shared by all three: **random masking at
training time, iterative unmasking at inference time**, with the loss
computed only on masked positions. This replaces the binary teacher-
forcing schedule with a continuous distribution over information-flow
patterns — every direction of the DAG is exposed to the model during
training because *any* subset of nodes may be masked.

## 5. Recommended action for v8 -> v9

### 5.1 Minimal change (drop-in)

Keep v8's StructuredHead + cross-attn target head. Replace §3.4
"Teacher forcing schedule" with **MoMask-style random masking on the
GT contact / phase signals**:

```
# At each batch:
for downstream_input in [gt_contact_for_target, gt_phase_for_support]:
    mask_ratio = uniform(0, 1)
    mask = bernoulli(mask_ratio, shape=(B, T))
    downstream_input[mask] = MASK_TOKEN  # or a learned mask embedding
```

This avoids the consistency problem of scheduled sampling because
the model learns to predict downstream tasks from *every* level of
upstream noise, not just GT and partial-rollout.

Expected effort: ~50 LOC change in `interaction_predictor.py` +
config flag. Direct path forward from v8.

### 5.2 Larger redesign (v9 spec)

Reformulate the four heads (contact, target, phase, support) as one
**joint masked-diffusion head**: input is the (B, T, 5+128+3+4)
multi-hot label tensor with random Bernoulli masking; output is
the unmasked-prediction logits at every position.

Training loss: weighted cross-entropy on masked positions (MDLM
recipe) **OR** SEDD score-entropy. Either gives a principled,
consistent estimator without the train-test gap.

Inference: K-step iterative unmasking (K=5-10) with cosine schedule
(MaskGIT/MoMask). The DAG order can be respected by *biasing the
unmask schedule*: unmask contact first, then phase / target, then
support. This recovers the inductive bias without changing the loss.

This is the v9 design. Start prototype after v8 results land.

## 6. References (full citations per CLAUDE.md "cite papers in full")

### 6.1 Primary candidates

- Guo, C., Mu, Y., Javed, M.G., Wang, S., Cheng, L. **"MoMask:
  Generative Masked Modeling of 3D Human Motions."** CVPR 2024.
  arXiv: 2312.00063.
  GitHub: EricGuo5513/momask-codes (1.3k stars).

- Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin,
  Y., Wen, J.-R., Li, C. **"Large Language Diffusion Models."**
  ICLR 2025 / ICML 2025. arXiv: 2502.09992.
  GitHub: ML-GSAI/LLaDA (3.8k stars).

- Chen, B., Monsó, D.M., Du, Y., Simchowitz, M., Tedrake, R.,
  Sitzmann, V. **"Diffusion Forcing: Next-Token Prediction Meets
  Full-Sequence Diffusion."** NeurIPS 2024.
  arXiv: 2407.01392. GitHub: buoyancy99/diffusion-forcing (1.2k stars).

- Ye, H., Xu, D. **"DiffusionMTL: Learning Multi-Task Denoising
  Diffusion Model from Partially Annotated Data."** CVPR 2024.
  arXiv: 2403.15389. GitHub: prismformore/DiffusionMTL.

- Huang, X., He, G., et al. **"Self Forcing: Bridging the Train-Test
  Gap in Autoregressive Video Diffusion."** NeurIPS 2025 Spotlight.
  arXiv: 2506.08009. GitHub: guandeh17/Self-Forcing (3.3k stars).

- Lou, A., Meng, C., Ermon, S. **"Discrete Diffusion Modeling by
  Estimating the Ratios of the Data Distribution."** ICML 2024
  *Best Paper*. arXiv: 2310.16834.
  GitHub: louaaron/Score-Entropy-Discrete-Diffusion (725 stars).

- Sahoo, S.S., Arriola, M., Gokaslan, A., Marroquin, E.M., Rush,
  A.M., Schiff, Y., Chiu, J.T., Kuleshov, V. **"Simple and Effective
  Masked Diffusion Language Models."** NeurIPS 2024.
  arXiv: 2406.07524. GitHub: kuleshov-group/mdlm.

### 6.2 Honourable mentions

- Arriola, M., Gokaslan, A., Chiu, J.T., Yang, Z., Qi, Z., Han, J.,
  Sahoo, S.S., Kuleshov, V. **"Block Diffusion: Interpolating Between
  Autoregressive and Diffusion Language Models."** ICLR 2025 *Oral*.
  arXiv: 2503.09573. GitHub: kuleshov-group/bd3lms.

- Yang, Y., et al. **"Multi-Task Dense Predictions via Unleashing the
  Power of Diffusion (TaskDiffusion)."** NeurIPS 2024 (under review,
  OpenReview ID: TzdTRC85SQ).
  GitHub: YuqiYang213/TaskDiffusion.

- Chang, H., Zhang, H., Jiang, L., Liu, C., Freeman, W.T. **"MaskGIT:
  Masked Generative Image Transformer."** CVPR 2022. arXiv: 2202.04200.
  Foundational; used as paradigm reference.

- Ning, M., Li, M., Su, J., Salah, A.A., Ertugrul, I.O. **"Elucidating
  the Exposure Bias in Diffusion Models."** ICLR 2024.
  arXiv: 2308.15321. GitHub: forever208/ADM-ES. Diagnostic only.

- Bengio, S., Vinyals, O., Jaitly, N., Shazeer, N. **"Scheduled
  Sampling for Sequence Prediction with Recurrent Neural Networks."**
  NeurIPS 2015. arXiv: 1506.03099. The method we are replacing.

- Huszár, F. **"How (not) to Train your Generative Model: Scheduled
  Sampling, Likelihood, Adversary?"** arXiv: 1511.05101, 2015.
  Original consistency critique of scheduled sampling.

### 6.3 Companion docs (this project)

- `analyses/2026-05-05_predictor_v8_design.md` (current v8 design
  using scheduled sampling; §3.4 is what this doc proposes to
  replace).
- `analyses/2026-05-05_v7fix_results_and_v6_baseline_correction.md`
  (the 21 cm L2 floor that motivated v8).

## 7. Decision tree

If v8 (current scheduled-sampling design) trains and lands within
acceptance gate (target_top1_token_recall ≥ 0.30, contact macro_f1
≥ 0.24, phase macro F1 ≥ 0.62, support macro F1 ≥ 0.40):

- **First fallback if downstream heads regress vs v7-fix**: §5.1
  minimal change — replace scheduled-sampling with MoMask-style
  random masking. Single-config v8.1 rerun.

- **Second fallback if §5.1 doesn't recover**: §5.2 v9 redesign with
  joint masked diffusion (LLaDA / MDLM recipe). Larger commit,
  ~1 week of engineering. Use SEDD score-entropy loss for principled
  small-class objective.

- **Third fallback if v9 doesn't recover within 4 weeks**: switch to
  Self-Forcing-style on-policy rollout training. Highest engineering
  cost, but most direct attack on the train-test gap if nothing else
  works.

If v8 lands cleanly, log this doc as "queued v9 work" — masked
diffusion is the right next step regardless of v8 outcome because it
also handles the **sparse-pseudo-label** problem (50 % contact
coverage) better than scheduled sampling.
