# 2026-05-05 — v8 Round 1 diagnosis + v8.1 plan (multi-hot binary GT)

## TL;DR

v8 Round 1 server retrain produced a **mixed** result, not a flat negative:

- **Pelvis target L2 +18 pp on <10cm hit** (33.9 → 51.9%) — architecture works
- **Hand / foot target L2 unchanged** — moving-contact gap, motion-stream KV needed (v9 candidate)
- **Phase macro F1 0.632 → 0.577**, support 0.397 → 0.378 — train-test gap from teacher forcing
- **target_top1_token_recall 0.093** — but the metric is wrong (single-point GT vs multi-token contact area)
- **Consistency loss reverse-trended up** — fixed-weight hinge ignored by optimizer

Three cited 2024-2025 papers prescribe direct fixes for three of the four
failures; the 4th (motion-stream KV) is deferred to v9. v8.1 is one
retrain (~6 h) testing all three fixes simultaneously. v9 is the
follow-up if v8.1 still doesn't close the hand/foot gap.

## 1. Per-failure root cause + literature prescription

### 1.1 Phase / support regression — TF train-test gap

**Diagnosis (data-driven)**: phase val_loss flat at 0.35 throughout
training while train drops 0.35 → 0.27. MLP head with `contact_emb`
input that is GT during epochs 0–50 and `pred(recall=35%)` during eval
→ severe distribution shift on the conditioning input.

**Literature**: scheduled sampling (Bengio NeurIPS 2015) was proven
**non-consistent** by Huszár (arXiv:1511.05101, 2015) — the train-test
gap is structural, not a tunable.

**2024-2026 SOTA**: random masking + iterative unmasking. The same
paradigm we already use for Stage B. Specifically:

- **MoMask** (Guo et al., CVPR 2024, arXiv:2312.00063, [github 1.3k★](https://github.com/EricGuo5513/momask-codes))
- **LLaDA** (Nie et al., ICLR 2025, arXiv:2502.09992, 3.8k★)
- **Diffusion Forcing** (Chen et al., NeurIPS 2024, arXiv:2407.01392, 1.2k★)
- **Self Forcing** (Huang et al., NeurIPS 2025 Spotlight, arXiv:2506.08009, 3.3k★)

Mechanism: at training time, sample a mask ratio `r ~ Uniform[0, 1]`
per batch, Bernoulli-mask `gt_contact` and `gt_phase` with that ratio,
fill masked positions with model predictions (or `[MASK]` token).
Loss flows through the head regardless of which positions are GT vs
pred. Provably consistent estimator: model trained on every information
mix, never sees a distribution it wasn't trained for.

### 1.2 target_top1 = 0.093 — wrong metric, wrong GT

**Diagnosis**: top-K lift ratio decreases (top-1 12×, top-3 9×, top-5
7× over random-128) → model has spatial concept but argmax bounces
between adjacent tokens. With FPS-128 token spacing ~ 0.088 m and KL
kernel σ = 0.08 m, GT is near-Dirac → small spatial errors → big top-1
miss.

**Bigger problem**: the user pointed out that hard one-hot GT is
inconsistent with HOI semantics. Palm contact covers a ~5 cm region;
multiple adjacent object tokens are all valid contact targets. Forcing
softmax to spike on a single token over-constrains the model.

**Literature confirms**: every HOI affordance paper surveyed uses
**multi-hot binary GT**, not one-hot, not soft:

| Paper | GT representation | Loss |
|---|---|---|
| EgoChoir (NeurIPS 2024) | per-vertex `{0,1}`, threshold = 5 cm SMPL-X | focal + dice + KL motion |
| Text2HOI (CVPR 2024) | per-FPS-point `{0,1}` | BCE + dice + KL latent |
| HOI-Diff APDM (CVPR 2025 WS) | 8-joint binary contact | BCE + MSE offset |

**Fix**: GT is binary mask over 128 tokens; loss is per-token sigmoid
focal BCE + dice. Position tolerance built into the GT itself via the
distance threshold, not into the loss kernel.

### 1.3 Consistency loss reverse-trended

**Diagnosis**: `loss_consistency` rose from 0.44 → 0.75 during training
(70 % growth) while weight = 0.1 means it contributed ~ 1.7 % of total.

**Literature**: this is a known pathology of the **fixed-weight penalty
method** for soft constraints (Bertsekas 1982 onward). Modern fix is
augmented Lagrangian with learnable multipliers — **Cooper** (Gallego-
Posada et al., NeurIPS 2025, arXiv:2504.01212, [github 158★](https://github.com/cooper-org/cooper)).

**v8.1 scope**: too heavy. Drop the constraint entirely. The DAG
conditioning via `contact_emb` / `phase_emb` already provides implicit
information flow; we don't need redundant explicit constraints. If
post-v8.1 evidence shows we need them back, do it via Cooper as v9
candidate.

### 1.4 Hand/foot moving-contact gap

**Diagnosis**: pelvis target L2 improved 15.4 → 14.4 cm (+18 pp on
<10 cm hit), hand/foot stayed at 22-26 cm. Pelvis is stationary during
contact; hand/foot move. The query `frame_q + part_query[p]` does not
have an explicit motion-aware feature.

**Literature**: **EgoChoir** (Yang et al., NeurIPS 2024, arXiv:2405.13659,
[github 30★](https://github.com/yyvhang/EgoChoir_release)) explicitly
addresses this with a parallel **motion-stream KV** alongside the
object-token KV, with τ-modulated gating. On their GIMO benchmark
(per-frame contact), this lifts hand/foot recall 3-5 pp vs single-stream.

**v8.1 scope**: deferred to v9 — architectural change, ~100 lines.
v8.1 first proves the loss + GT change works; v9 adds motion KV if
needed.

## 2. v8.1 design — three changes, one retrain

### Change A: Random Bernoulli mask replaces teacher forcing

In `StructuredHead.forward`, replace the binary TF decision with:

```python
# Per-batch mask ratio r ~ Uniform[0, 1]
mask_ratio = torch.rand(1, device=x.device).item()
# Bernoulli mask per (B, T, P) cell: 1 = use GT, 0 = use pred
mask = torch.bernoulli(torch.ones_like(gt_contact) * mask_ratio)
contact_for_downstream = mask * gt_contact + (1 - mask) * contact_prob
```

Training-only path. At eval, contact_for_downstream is always
contact_prob (no GT seen). The head learns to handle every mix
in `[0, 1]` from epoch 0; no anneal schedule needed.

Same scheme for `phase_for_downstream` using `gt_phase_one_hot` vs
`phase_prob`.

### Change B: Multi-hot binary GT + focal + dice + sigmoid output

**Path B selected** (user 2026-05-05): drop the softmax-xyz back-compat
output entirely. Stage B's InteractionTokenizer must be refactored in
v8.1b to consume `contact_target_attn (B, T, 5, M)` directly (replaces
the current 15-channel `contact_target_xyz` flatten in
`interaction_tokenizer.py:255`). Predictor v8.1a ckpt is not
Stage-B-runnable until v8.1b lands.

In `StructuredHead.forward`, change target attention to emit logits:

```python
attn_logits = self.target_attn(q_flat, obj_tokens)         # (B, T*P, M) — pre-sigmoid
attn_logits = attn_logits.reshape(B, T, P, M)

# v8.1 primary output: per-token sigmoid (each token independent [0,1])
# No softmax, no back-compat xyz output. Stage B v8.1b will consume
# contact_target_attn directly.
target_attn_pred = sigmoid(attn_logits)
```

The `CrossAttentionWeightsOnly` module needs a `softmax_output: bool =
True` flag — old default keeps the v8 behaviour, v8.1 sets `False` to
return logits.

In `PredictorLoss.forward`, replace the KL path:

```python
# Build multi-hot GT: tokens within tau of GT closest_xyz are positive
diff = gt_xyz.unsqueeze(-2) - obj_xyz.view(B, 1, 1, M, 3)
d = diff.norm(dim=-1)                                       # (B, T, P, M)
tau = tensor([0.05, 0.05, 0.03, 0.03, 0.10])                # hand, hand, foot, foot, pelvis
gt_mask = (d < tau.view(1, 1, P, 1)).float()                # multi-hot

# Focal BCE
bce = F.binary_cross_entropy_with_logits(logits, gt_mask, reduction='none')
p_t = pred * gt_mask + (1 - pred) * (1 - gt_mask)
focal = bce * (1 - p_t).pow(focal_gamma)                    # γ = 2.0
alpha_t = alpha * gt_mask + (1 - alpha) * (1 - gt_mask)     # α = 0.25
loss_focal = (alpha_t * focal).mean(dim=-1)                 # (B, T, P)

# Dice on the multi-hot mask
inter = (pred * gt_mask).sum(dim=-1)
loss_dice = 1.0 - 2 * (inter + 1e-6) / (pred.sum(-1) + gt_mask.sum(-1) + 1e-6)

loss_target = 0.5 * loss_focal + 0.5 * loss_dice
```

Gate by `contact > threshold` as before — only contact-positive cells
contribute (closest-mesh-point is geometrically defined for non-contact
cells but doesn't represent actual contact regions).

### Change C: Drop consistency loss

Delete the 4 hinge terms. Set `consistency_weight: 0.0` in v8.1 config.
Keep the `_consistency_loss` function dormant in `losses.py` for v9
revival via Cooper.

### Pseudo-label extraction: NO change

The npz files keep `contact_target_xyz_gt: (T, 5, 3)`. Multi-hot mask
is built per-batch at training time using current FPS sample. Saves
re-extracting 8475 sequences. The v12_strict pipeline is unchanged.

## 3. Eval metric update — IoU + F1 on multi-hot mask

Current `target_top1_token_recall` is now an artifact metric (single-
token argmax doesn't represent multi-token contact regions). Add the
correct ones:

```python
# Per contact-positive (frame, part) cell:
mask_pred = (pred > 0.5)                                    # (B, T, P, M)
mask_gt = (d < tau).float()
intersection = (mask_pred & mask_gt).sum(-1)
union = (mask_pred | mask_gt).sum(-1)
iou = intersection / max(union, 1)                          # (B, T, P)
precision = intersection / max(mask_pred.sum(-1), 1)
recall = intersection / max(mask_gt.sum(-1), 1)
f1 = 2 * precision * recall / max(precision + recall, ε)
```

Aggregate over contact-positive cells → `mean_iou_contact_cells`,
`mean_f1_contact_cells`.

Keep top-K recall computed for direct comparison to v8 numbers, but
mark it deprecated.

## 4. v8.1 acceptance gate (vs v8 best, ep34)

| metric | v7-fix | v8 best | v8.1 target | gate |
|---|---:|---:|---|---|
| target IoU on contact cells (NEW primary) | n/a | n/a | ≥ 0.30 | hard |
| target F1 on contact cells (NEW) | n/a | n/a | ≥ 0.40 | hard |
| target xyz L2 (back-compat) | 21.77 | 21.55 | ≤ 19 | soft |
| target_top1_token_recall (DEPRECATED) | n/a | 0.093 | n/a | not gated |
| contact macro_f1 | 0.237 | 0.235 | ≥ 0.24 | hard |
| phase macro F1 | 0.632 | 0.577 | ≥ 0.62 | hard |
| support macro F1 | 0.397 | 0.378 | ≥ 0.40 | hard |
| pelvis target L2 (v8 win to keep) | 15.4 | 14.4 | ≤ 14.5 | hard |

**Pass condition**: 5/6 hard gates pass + pelvis advantage retained.

Failure routes:
- **phase / support still regress** → TF wasn't dominant; v9 simplifies
  heads (drop contact_emb input from phase head, etc.)
- **IoU / F1 fail but pelvis still wins** → motion-stream KV (EgoChoir)
  is the next architectural change → v9
- **Pelvis advantage lost** → revert; the multi-hot GT broke pelvis;
  unlikely (pelvis already converges sharply)

## 5. v9 backlog (conditional on v8.1 evidence)

If v8.1 hits IoU ≥ 0.30 but hand/foot per-part still 22-26 cm:
- **EgoChoir motion-KV stream** ([code](https://github.com/yyvhang/EgoChoir_release)),
  ~100 lines

If v8.1 phase / support still regress:
- **TaskPrompter task tokens** (Ye & Xu, ICLR 2023, arXiv:2303.00748,
  [code 327★](https://github.com/prismformore/Multi-Task-Transformer)),
  ~150 lines

If overall MTL gradient interference shows up:
- **DTME-MTL** (Jeong et al., ICCV 2025, arXiv:2507.07485) for
  transformer-trunk gradient surgery
- **Aligned-MTL** (Senushkin et al., CVPR 2023) cheaper alternative

If consistency constraints become provably needed:
- **Cooper augmented Lagrangian** (Gallego-Posada et al., NeurIPS 2025,
  arXiv:2504.01212, [code 158★](https://github.com/cooper-org/cooper))

## 6. Implementation file plan

| file | change | lines |
|---|---|---:|
| `src/piano/models/interaction_predictor.py` | `CrossAttentionWeightsOnly` softmax flag; `StructuredHead` mask-instead-of-TF; sigmoid output + softmax xyz side-output | +50 |
| `src/piano/training/losses.py` | `_focal_dice_target_loss` helper; `target_loss_kind="focal_dice"` path | +60 |
| `src/piano/training/train_predictor.py` | gate downstream gt feed by `mask_ratio` instead of TF flag; pass `gt_contact` and `gt_phase` always (head decides mask) | +30 |
| `scripts/stage_a_predictor/eval_predictor.py` | IoU / F1 / precision / recall on multi-hot mask | +50 |
| `tests/test_structured_head.py` | new tests: multi-hot GT construction, focal+dice ≥ 0, mask training reproduces TF=0 case, IoU metric correctness | +120 |
| `configs/training/predictor_v8_1_masked.yaml` | new config | +160 |

Total: ~470 lines.

## 7. References

- Bengio et al. NeurIPS 2015 — scheduled sampling. arXiv:1506.03099.
- Huszár arXiv:1511.05101 (2015) — scheduled sampling not consistent.
- Guo et al. CVPR 2024 — MoMask. arXiv:2312.00063.
- Yang et al. NeurIPS 2024 — EgoChoir. arXiv:2405.13659.
- Cha et al. CVPR 2024 — Text2HOI. arXiv:2404.00562.
- Peng et al. CVPR 2025 WS — HOI-Diff APDM. arXiv:2312.06553.
- Gallego-Posada et al. NeurIPS 2025 — Cooper. arXiv:2504.01212.
- Ye & Xu ICLR 2023 — TaskPrompter. arXiv:2303.00748.
- Jeong et al. ICCV 2025 — DTME-MTL. arXiv:2507.07485.
- Companion analyses:
  - `analyses/2026-05-02_alternatives_to_scheduled_sampling.md`
  - `analyses/2026-05-02_hoi_affordance_sota_survey_post_move_as_you_say.md`
  - `analyses/2026-05-02_mtl_dag_research_survey.md`
- Predecessor: `analyses/2026-05-05_predictor_v8_design.md`
