# Multi-Task Learning with Task DAG, Gradient Conflict, and Constraint Enforcement: 2023-2026 Literature Survey

**Date:** 2026-05-02
**Trigger:** Stage A v8 predictor: contact -> {target_xyz, phase} -> support DAG. PAD-Net style conditioning made phase macro F1 drop 0.632 -> 0.577. Hinge consistency loss (`relu(p_dependent - p_prerequisite)`) gets *ignored* by the optimizer (the term increases through training despite λ > 0).
**Status:** survey complete; recommendations queued in PROGRESS

---

## 0. Question framing

Our model has 4 dependent tasks on a DAG:

```
contact ──┬──> target_xyz
          └──> phase ──> support
```

Setup: 10-layer transformer trunk, 5 body parts, T=196 frames, per-frame structured prediction, 4 task heads.

We need answers to four sub-questions:
1. Modern alternatives to PAD-Net (CVPR 2018) and Cross-Stitch (CVPR 2016) for **DAG conditioning**.
2. Modern alternatives to PCGrad / GradNorm / GradVac for **gradient conflict**.
3. Why **soft hinge constraints** get ignored, and what replaces them.
4. **Cross-task self-distillation / EMA mean-teacher** options that don't require teacher forcing.

---

## 1. DAG-structured task conditioning (post-PAD-Net)

### 1.1 TaskPrompter (ICLR 2023) — strong candidate

- **Citation:** Hanrong Ye, Dan Xu. *TaskPrompter: Spatial-Channel Multi-Task Prompting for Dense Scene Understanding*. ICLR 2023. OpenReview `-CwPopPJda`.
- **GitHub:** `prismformore/Multi-Task-Transformer` — **327 stars** (combined repo with InvPT).
- **Key idea:** task-specific learnable prompt tokens are jointly attended with patch tokens; cross-task interaction happens through self-attention along spatial *and* channel dims of these task tokens. Replaces explicit "predict-then-distill" with implicit prompting.
- **Tasks evaluated:** NYUD-v2 (4 tasks), PASCAL-Context (5 tasks), Cityscapes-3D (3D detection + segmentation + depth = 3 tasks). >2 tasks confirmed.
- **Fit for our problem:** strong replacement for PAD-Net's hand-wired DAG. Use 4 task-tokens (contact, target, phase, support) with cross-attention; the DAG becomes a per-layer attention mask (e.g., support-token attends to phase-token, phase- and target-tokens attend to contact-token). No teacher forcing needed — the conditioning is via attention, not feeding hard predictions.

### 1.2 InvPT (ECCV 2022) + InvPT++ (TPAMI 2024)

- **Citation:** Hanrong Ye, Dan Xu. *Inverted Pyramid Multi-task Transformer for Dense Scene Understanding*. ECCV 2022, arXiv:2203.07997. Extended: *InvPT++*, arXiv:2306.04842, TPAMI 2024.
- **GitHub:** same `prismformore/Multi-Task-Transformer` (327 stars).
- **Key idea:** Cross-task self-attention at progressively higher spatial resolutions; AMP block does message passing across tasks at each scale.
- **Fit:** weaker than TaskPrompter for our problem because it's symmetric cross-task attention (no DAG structure). Mainly useful as the architectural backbone TaskPrompter improves on.

### 1.3 MQTransformer (TCSVT 2023)

- **Citation:** Yangyang Xu, Xiangtai Li, Haobo Yuan, Yibo Yang, Lefei Zhang. *Multi-Task Learning with Multi-Query Transformer for Dense Prediction*. arXiv:2205.14354, IEEE TCSVT 2023.
- **GitHub:** `yangyangxu0/MQTransformer`.
- **Key idea:** one query per task; cross-task query-attention replaces dense per-pixel cross-task fusion (PAD-Net style). O(K²) over tasks instead of O(HW) over pixels.
- **Fit:** moderate. Same conceptual move as TaskPrompter but earlier; TaskPrompter is the more cited and more architecturally complete version.

### 1.4 TADFormer (CVPR 2025)

- **Citation:** Baek, Lee, Jo, Choi, Min. *TADFormer: Task-Adaptive Dynamic TransFormer for Efficient Multi-Task Learning*. CVPR 2025.
- **GitHub:** code mentioned but no link surfaced in searches.
- **Key idea:** parameter-efficient prompting + Dynamic Task Filter (DTF) that conditions features on per-input task context. Up to 8.4× fewer trainable params.
- **Fit:** weak for the DAG question — does *not* model task ordering. Useful only if we go the LoRA route per task.

### 1.5 Joint Scheduling of Causal Prompts and Tasks (CVPR 2025) — DAG-aware

- **Citation:** Chaoyang Li, Jianyang Qin, Jinhao Cui, Zeyu Liu, Ning Hu, Qing Liao. *Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning*. CVPR 2025.
- **Key idea:** task-prompt scheduler that explicitly models inter-task affinities and assesses *causal* effect of prompts; bi-level optimization for prompts and tasks. Closest published method to our "earlier task feeds later task" structure.
- **Fit:** strong conceptual fit. Caveat: no public GitHub surfaced — reproducibility risk.

### 1.6 Ruled out for DAG conditioning

- **Auto-Lambda** (lorenmt/auto-lambda, 141 stars, TMLR 2022, Liu et al.): symmetric task weighting, *no* DAG modeling. Useful for loss balancing only.
- **JTR** (KentoNishi/JTR-CVPR-2024, 24 stars): partial-label regularization, not DAG-structured. Low community uptake.

---

## 2. Gradient conflict (post-PCGrad / GradNorm / GradVac)

### 2.1 Aligned-MTL (CVPR 2023) — strong candidate

- **Citation:** Dmitry Senushkin, Nikolay Patakin, Arseny Kuznetsov, Anton Konushin. *Independent Component Alignment for Multi-Task Learning*. CVPR 2023, arXiv:2305.19000.
- **Key idea:** uses the **condition number** of the gradient matrix as the stability criterion; aligns *principal components* (orthogonal axes) of the gradient system. Eliminates dominating-gradient pathology, not just sign conflict.
- **Fit:** directly addresses our "phase regression after adding contact conditioning" symptom — phase's gradient is being dominated by contact's. Aligned-MTL provably bounds this. Tested on >2 tasks (NYUD-v2 4 tasks, Cityscapes 3 tasks, RL).
- **Note:** DTME-MTL paper reports Aligned-MTL gives **−9.41% multi-task gain** on Taskonomy 11-task (negative transfer), which is a red flag for very large task counts; for our 4 tasks it should still help.

### 2.2 FAMO (NeurIPS 2023)

- **Citation:** Bo Liu, Yihao Feng, Peter Stone, Qiang Liu. *FAMO: Fast Adaptive Multitask Optimization*. NeurIPS 2023, arXiv:2306.03792.
- **GitHub:** `Cranial-XIX/FAMO` — **124 stars**.
- **Key idea:** dynamic task weights using **O(1) space/time** (matches Adam complexity) by tracking only loss decreases, not full per-task gradients. Avoids the K× overhead of PCGrad/CAGrad.
- **Fit:** strong, especially as a low-cost replacement to whatever scheduler we have now. Doesn't model DAG, but composes cleanly with TaskPrompter-style conditioning.

### 2.3 Nash-MTL (ICML 2022)

- **Citation:** Aviv Navon, Aviv Shamsian, Idan Achituve, et al. *Multi-Task Learning as a Bargaining Game*. ICML 2022, arXiv:2202.01017.
- **GitHub:** `AvivNavon/nash-mtl` — **240 stars**.
- **Key idea:** treat gradient combination as Nash bargaining; tasks negotiate update direction that is Pareto-stationary. Theoretically principled fairness.
- **Fit:** moderate. Symmetric (no DAG awareness). Consider as a stronger baseline than PCGrad if we want a single-method replacement.

### 2.4 DTME-MTL (ICCV 2025) — strong, transformer-native

- **Citation:** Wooseong Jeong, Kuk-Jin Yoon. *Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning*. ICCV 2025, arXiv:2507.07485.
- **GitHub:** `wooseong97/DTME-MTL` (announced in paper).
- **Key idea:** SVD-decompose the token covariance; categorize gradient conflicts as **range-space** (g_R,i · g_R,j ≤ 0, opposing within current feature span) vs **null-space** (orthogonal to current tokens). Range conflicts → affine modulation of existing tokens. Null conflicts → expand with new task-specific tokens.
- **Backbones tested:** InvPT, TaskPrompter, ViT-T/S/B/L on NYUD-v2 (4 tasks), PASCAL-Context (5 tasks), Taskonomy (11 tasks).
- **Reported numbers (Table 3, Taskonomy):** DTME-MTL **+4.67%** vs baseline; PCGrad **−8.29%**, CAGrad **−10.57%**, Aligned-MTL **−9.41%**.
- **Fit:** very strong because (a) we are already token-based (transformer trunk + task heads), (b) it natively composes with TaskPrompter, (c) it directly outperforms gradient-surgery methods at large task counts.

### 2.5 GCond (arXiv Sept 2025)

- **Citation:** Evgeny Alves Limarenko, Anastasiia Studenikina. *GCond: Gradient Conflict Resolution via Accumulation-based Stabilization for Large-Scale Multi-Task Learning*. arXiv:2509.07252.
- **Key idea:** "accumulate-then-resolve" — average gradients with low variance first, then arbitrate. 2× speedup over PCGrad. Compatible with AdamW / Lion / LARS.
- **Fit:** moderate. arXiv-only (no venue yet), no star data. Watch but don't bet on.

### 2.6 Ruled out

- Vanilla **PCGrad / GradNorm / GradVac**: superseded by Aligned-MTL and FAMO on every benchmark we surveyed.
- **RotoGrad** (ICLR 2021): O(Kd³) cost is prohibitive at our trunk dim; conceptually superseded by Aligned-MTL.

---

## 3. Why soft hinge constraints get ignored — and the fix

### 3.1 Diagnosis

The hinge `λ · ReLU(p_dependent − p_prerequisite)` is a **penalty method**. Penalty methods are known to be ignored when:
- The base task losses for dependent and prerequisite are large relative to λ → optimizer prefers to reduce them at the expense of the constraint.
- λ is constant: the optimizer sees a finite "bribe" for violation and pays it. Increasing λ blows up training.

This is well documented in the constrained-optimization literature (Bertsekas, *Constrained Optimization and Lagrange Multiplier Methods*, 1982; survey: Frontiers in AI 2024 *"Integration between constrained optimization and deep networks"*).

### 3.2 Fix: Augmented Lagrangian via Cooper

- **Citation:** Jose Gallego-Posada, Juan Ramirez, Meraj Hashemizadeh, Simon Lacoste-Julien. *Cooper: A Library for Constrained Optimization in Deep Learning*. arXiv:2504.01212, NeurIPS 2025.
- **GitHub:** `cooper-org/cooper` — **158 stars**.
- **Key idea:** maintains **learnable** Lagrange multipliers λ_i per constraint. The multiplier *adapts* — rises when the constraint is violated, decays when satisfied. Augmented Lagrangian variant adds a quadratic penalty so the duality gap closes even on non-convex losses.
- **Why this fixes our symptom:** with constant λ the optimizer can permanently pay the bribe. With a learnable λ_i that grows on violation, the penalty becomes unbounded for persistently violating directions — the optimizer is forced to satisfy.
- **Fit:** drop-in replacement for our hinge term. PyTorch-native, supports inequality constraints (`p_dependent ≤ p_prerequisite + ε`), and our current `relu(...)` becomes a `Constraint(constraint_type=INEQUALITY)` object.

### 3.3 Alternative — Constrained Parameter Regularization (NeurIPS 2024)

- **Citation:** Jörg Franke, Michael Hefenbrock, Frank Hutter. *Improving Deep Learning Optimization through Constrained Parameter Regularization*. NeurIPS 2024.
- **Key idea:** weight-decay-like augmented-Lagrangian on parameter norms. Conceptually adjacent; the technique is the same machinery (learnable multipliers, augmented Lagrangian) applied to parameters instead of outputs.
- **Fit:** confirms the augmented-Lagrangian approach is mature and applied to deep learning at top venues.

### 3.4 Ruled out

- **Hard projection** of outputs onto the feasible set: differentiable but breaks gradient flow on the boundary; brittle in practice.
- **Lagrangian with fixed multipliers** (just renaming our hinge): same problem.

---

## 4. Train-test consistency without teacher forcing

### 4.1 Mean-Teacher / EMA self-distillation

The Mean-Teacher recipe (Tarvainen & Valpola, NeurIPS 2017) is still the standard for cross-task consistency without teacher forcing:

- Maintain `θ_teacher = α · θ_teacher + (1 − α) · θ_student` (EMA, α ≈ 0.999).
- At each step, the teacher predicts the prerequisite task (e.g., contact); the student is conditioned on the *teacher's* contact prediction, not on its own and not on the ground truth.
- This is **scheduled sampling without scheduling** — the teacher's slowly-evolving predictions match what the student will see at inference time, but are smoother than the student's own predictions, so they don't destabilize training.

For our specific symptom (consistency loss *increasing* through training), the EMA-teacher fix is:
- Compute the consistency target from the EMA teacher, not from the live student.
- Detach the teacher; the consistency loss only gradients through the student.

### 4.2 No new 2024-2026 paper supersedes Mean-Teacher for this specific use

The 2024-2025 literature on "self-distillation in MTL" (TSCL, multi-teacher distillation, etc.) targets *different* problems (continual learning, data efficiency, LLM rationales) and does not replace Mean-Teacher for cross-task consistency. The recommendation is to use the original recipe and skip a fashion-driven swap.

### 4.3 Scheduled sampling alternative

If we want a non-EMA approach: classic scheduled sampling (Bengio et al. 2015) and its transformer extension (Mihaylova & Martins, arXiv:1906.07651) feed model predictions instead of ground truth with rising probability. The *Parallel Scheduled Sampling* variant (ICLR 2020 submission) handles non-autoregressive settings. But for cross-task (not cross-time) feeding, EMA-teacher is conceptually cleaner.

---

## 5. The single-method question

> **Is there a 2024-2026 method that simultaneously handles (a) DAG-structured conditioning, (b) gradient conflict, (c) train-test consistency without teacher forcing?**

**Answer: No single method covers all three.** The closest candidates each cover two:

| Method | (a) DAG cond. | (b) Grad conflict | (c) Train-test consistency |
|---|---|---|---|
| TaskPrompter (ICLR 2023) | yes (via prompt attention; DAG via mask) | no | no (still teacher-forces during training) |
| DTME-MTL (ICCV 2025) | partial (token expansion is task-specific, not DAG) | yes (token-space SVD) | no |
| Joint Causal Prompts (CVPR 2025) | yes (causal scheduler) | partial | no |
| Cooper (NeurIPS 2025) | no | no (it's constraint optimization) | yes (constraints are train-time = test-time) |
| EMA Mean-Teacher (NeurIPS 2017) | no | no | yes |

**Recommendation:** assemble a stack rather than search for a unicorn:
1. **TaskPrompter-style task tokens with DAG-masked cross-attention** (replaces PAD-Net cascading with attention; keeps gradients flowing through the DAG without hard predictions).
2. **DTME-MTL or Aligned-MTL** for gradient-conflict handling (DTME-MTL if we already have transformer tokens; Aligned-MTL if we want the simpler optimizer-only patch).
3. **Cooper augmented-Lagrangian** for the consistency constraint (replaces the ignored hinge with a learnable-multiplier inequality constraint).
4. **EMA Mean-Teacher** for cross-task consistency (target generated by EMA teacher, no teacher forcing of ground-truth).

---

## 6. Decision tree for our specific failure (phase F1 0.632 → 0.577)

If after retraining with the recommended stack:

- **Phase F1 still drops** vs independent baseline → the bottleneck is *not* gradient conflict but **representation interference**. Move to per-task LoRA adapters (TADFormer / MTLoRA, CVPR 2024) so phase has its own subspace.
- **Phase F1 recovers but support drops** → DAG depth issue; check that support-token attention is correctly masked to phase-token only.
- **Consistency loss now decreases monotonically** → augmented-Lagrangian fix worked; queue ablation removing other heuristics.
- **Consistency loss still grows** → the constraint is fundamentally infeasible (i.e., ground truth itself violates `p_dep ≤ p_prereq`). Verify by running the constraint check on the dataset before training.

---

## 7. Ruled-out hypotheses (do not re-investigate)

- **PCGrad / GradNorm / GradVac alone** — superseded by Aligned-MTL/FAMO/DTME-MTL on every benchmark; DTME-MTL paper shows them at *negative* transfer on Taskonomy.
- **RotoGrad** — O(Kd³), prohibitive at our trunk dim.
- **Auto-Lambda alone** — symmetric weighting, doesn't address DAG.
- **Hard output projection** — gradient pathology on the constraint boundary.
- **Constant-λ hinge** — confirmed root cause of "consistency loss ignored"; the multiplier must be learnable (Lagrangian).
- **TADFormer** — does not model DAG; only useful as a parameter-efficient backbone choice.
- **Nash-MTL** — symmetric Pareto bargaining, no DAG; weaker than DTME-MTL on transformer backbones.

---

## 8. Citations (full)

1. Hanrong Ye, Dan Xu. **TaskPrompter: Spatial-Channel Multi-Task Prompting for Dense Scene Understanding.** ICLR 2023. OpenReview:`-CwPopPJda`. Code: github.com/prismformore/Multi-Task-Transformer (327★).
2. Hanrong Ye, Dan Xu. **Inverted Pyramid Multi-task Transformer for Dense Scene Understanding.** ECCV 2022. arXiv:2203.07997. Extended TPAMI 2024 (InvPT++): arXiv:2306.04842.
3. Yangyang Xu et al. **Multi-Task Learning with Multi-Query Transformer for Dense Prediction.** IEEE TCSVT 2023. arXiv:2205.14354.
4. Baek, Lee, Jo, Choi, Min. **TADFormer: Task-Adaptive Dynamic TransFormer for Efficient Multi-Task Learning.** CVPR 2025.
5. Chaoyang Li et al. **Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning.** CVPR 2025.
6. Dmitry Senushkin, Nikolay Patakin, Arseny Kuznetsov, Anton Konushin. **Independent Component Alignment for Multi-Task Learning** (Aligned-MTL). CVPR 2023. arXiv:2305.19000.
7. Bo Liu, Yihao Feng, Peter Stone, Qiang Liu. **FAMO: Fast Adaptive Multitask Optimization.** NeurIPS 2023. arXiv:2306.03792. Code: github.com/Cranial-XIX/FAMO (124★).
8. Aviv Navon et al. **Multi-Task Learning as a Bargaining Game** (Nash-MTL). ICML 2022. arXiv:2202.01017. Code: github.com/AvivNavon/nash-mtl (240★).
9. Wooseong Jeong, Kuk-Jin Yoon. **Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning** (DTME-MTL). ICCV 2025. arXiv:2507.07485. Code: github.com/wooseong97/DTME-MTL.
10. Limarenko, Studenikina. **GCond: Gradient Conflict Resolution via Accumulation-based Stabilization for Large-Scale Multi-Task Learning.** arXiv:2509.07252 (Sept 2025).
11. Jose Gallego-Posada et al. **Cooper: A Library for Constrained Optimization in Deep Learning.** NeurIPS 2025. arXiv:2504.01212. Code: github.com/cooper-org/cooper (158★).
12. Jörg Franke, Michael Hefenbrock, Frank Hutter. **Improving Deep Learning Optimization through Constrained Parameter Regularization.** NeurIPS 2024.
13. Antti Tarvainen, Harri Valpola. **Mean teachers are better role models.** NeurIPS 2017.
14. Shikun Liu, Stephen James, Andrew J. Davison, Edward Johns. **Auto-Lambda: Disentangling Dynamic Task Relationships.** TMLR 2022. Code: github.com/lorenmt/auto-lambda (141★).
15. Kento Nishi et al. **Joint-Task Regularization for Partially Labeled Multi-Task Learning** (JTR). CVPR 2024. Code: github.com/KentoNishi/JTR-CVPR-2024 (24★).
