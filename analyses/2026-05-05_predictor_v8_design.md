# 2026-05-05 — Stage A v8 design: affordance-style target + structured DAG heads

## TL;DR

v8 redesigns the predictor head along two complementary axes:

1. **Representation change** (motivated by Move-as-You-Say CVPR 2024 +
   v7-fix's 21 cm L2 architectural floor): replace per-part xyz regression
   with **soft attention over the 128 object tokens**, exactly the
   "affordance map" intermediate representation the paper champions.
2. **Architecture change** (motivated by reading the actual pseudo-label
   extraction DAG): replace 4 independent linear heads with **sequential
   conditioning that mirrors the extraction DAG**: contact → {target,
   phase} → support, plus an auxiliary consistency loss to enforce the
   physical priors that make labels self-consistent.

Combined, v8 fixes head-architecture issues W1 (target head too thin)
and W2 (head loses object identity) — see
`analyses/2026-05-05_predictor_quality_assessment_and_v8_design.md` if
that companion doc was filed; otherwise see Section 3.2 of this doc —
**without** changing the trunk (10-layer transformer, d=384, 6 heads,
ffn=1024).

Expected metric impact (v7-fix baseline → v8 prediction):

| metric | v7-fix | v8 (predicted) |
|---|---:|---:|
| target top-1 token recall | n/a | 35-50 % |
| target top-3 token recall | n/a | 65-80 % |
| target attention-weighted L2 (legacy comparison) | 21.77 cm | 8-15 cm |
| contact macro_f1 | 0.237 | 0.27-0.32 |
| phase macro F1 | 0.632 | 0.65-0.70 |
| support macro F1 | 0.397 | 0.45-0.52 |
| Stage B γ_int convergence | 0.02 | 0.05-0.10 |

## 1. Context: why v7-fix is not enough for z_int

`analyses/2026-05-05_v7fix_results_and_v6_baseline_correction.md`
established that v7-fix's 21.77 cm target overall L2 is the
architecture's normal performance, not a regression. v6 baseline is
21.13 cm on the same metric.

But as a **z_int source for Stage B**, the predictor is still poor:

- `contact recall (any_part) = 0.353`: 65 % of GT contact moments are
  silently dropped from z_int
- `target xyz overall L2 = 21.77 cm`: predicted contact location is
  often on the wrong patch / wrong face of the object (per-object
  breakdown shows large objects 25-44 cm L2)
- `target <5cm hit = 3.6 %`: nearly never within manipulation tolerance
- `foot recall = 0.000`: support-related contacts not propagated

Independent evidence that Stage B has noticed this and routed around it:
γ_int (the IntXAttn zero-init gate that scales z_int's contribution to
the generator) **converges to 0.02 across v4–v16**, vs typical
ControlNet-style gates 0.5–1.0 (`analyses/2026-05-01_v17_diagnostics_and_gumbel.md`,
`analyses/2026-05-03_gamma_int_re_evaluation.md`). Stage B is using z_int
at ~ 1/25 of typical control-conditioning weight — voting with its
weights that the signal isn't trustworthy.

Conclusion: **predictor quality is the dominant Stage B bottleneck**.
Per-step inference, training-data work (v12 strict), and Stage B
architecture tweaks are second-order until z_int actually carries useful
information.

## 2. Two diagnoses, one fix

### 2.1 Diagnosis A — Wrong output representation

`contact_target_xyz_gt` lives in **world coordinates** (closest mesh
point per body part per frame). Output range:

- chairs subset object footprint: ~1-2 m
- omomo subset large boxes / clothes stands: ~1-3 m
- neuraldome subset desks / sofas / pillows: ~2-3 m
- room-frame variance across clips: 5 m × 5 m × 2 m

Predicting world xyz with a single `nn.Linear(384, 15)` projection asks
the network to map a 384-d feature into this large manifold. 21 cm L2
is ~ 4 % of the dominant axis — plausible for a small head but
inadequate for grasp tasks where 2-5 cm is the relevant tolerance.

Per-object L2 directly confirms scale-coupling: small objects 10-12 cm,
large chairs 25-44 cm. Output range scales with object size, which is
the canonical symptom of a world-coordinate output without
object-frame anchor.

**Move as You Say (Wang et al., CVPR 2024)** solves the same problem
at scene scale by predicting a per-scene-point distance heatmap rather
than xyz coordinates. The output domain becomes "which point" (discrete
distribution over PC) rather than "where in space" (continuous
unbounded regression).

**Fix**: emit attention over the 128 object tokens
`(B, T, 5_parts, 128)`, supervised by Gaussian-kerneled GT distribution.

### 2.2 Diagnosis B — Independent heads ignore extraction DAG

`extract_*.py` builds labels along this DAG:

```
joints + obj_pose + obj_mesh
    │
    ├──▶ contact_state (T, 5)              [base; geometric + kinematic]
    │       │
    │       ├──▶ contact_target_xyz_gt     [extract_target.py:150 — only computes for contact frames]
    │       │
    │       └──▶ phase                      [extract_phase.py:122 uses any_contact_score]
    │               │
    │               └──▶ support            [extract_support.py:184: phase_stable gates hand_support]
```

Concrete cross-references in source:

- `target` only meaningful when contact > 0
  ([extract_target.py:150](src/piano/data/pseudo_labels/extract_target.py#L150))
- `phase` uses contact transitions
  ([extract_phase.py:122](src/piano/data/pseudo_labels/extract_phase.py#L122))
- `hand_support ⊂ phase==stable_contact ∩ hand_contact ∩ pelvis_stationary`
  ([extract_support.py:184-202](src/piano/data/pseudo_labels/extract_support.py#L184-L202))
- `sitting ⊂ pelvis_contact ∩ pelvis_stationary ∩ object_below_pelvis`
  ([extract_support.py](src/piano/data/pseudo_labels/extract_support.py))

In contrast, model heads
([interaction_predictor.py:317-320](src/piano/models/interaction_predictor.py#L317-L320)):

```python
self.contact_head = nn.Linear(d_model, num_body_parts)        # 384 → 5
self.target_head  = nn.Linear(d_model, num_body_parts * 3)    # 384 → 15
self.phase_head   = nn.Linear(d_model, num_phases)            # 384 → 3
self.support_head = nn.Linear(d_model, num_support_states)    # 384 → 4
```

All four are parallel projections from the same trunk feature `x`. They
must independently rediscover the cross-task structure that the
extractor encodes by hand. There is no explicit information flow
between heads.

Multi-task literature on this:

- **PAD-Net** (Xu et al., CVPR 2018): shallow-task predictions →
  deep-task input; depth + segmentation + normal estimation.
- **Cross-Stitch Networks** (Misra et al., CVPR 2016): layer-wise
  task-feature mixing; symmetric (PIANO is asymmetric DAG).
- **MTI-Net** (Vandenhende et al., ECCV 2020): explicit task-task
  message passing; heavy.
- **HOI Detection** (Liao et al., 2020 + later): verb head conditioned
  on object head — closest analog for our DAG.
- **SMPLify-X** (Pavlakos et al., CVPR 2019): hand/face/body sequential
  refinement at inference time.

PAD-Net + HOI Detection are the right anchors: predict the prerequisite
task first, embed its prediction, feed it as context to the dependent
task.

**Fix**: introduce a `StructuredHead` that runs heads in dependency
order (contact → {target, phase} → support), with each downstream head
seeing the previous heads' (embedded) predictions.

### 2.3 Why merge: the two diagnoses are entangled

The `target` head is the worst-affected by both diagnoses:

- W1 (head too thin — single 384→15 Linear) means it can't span the
  output manifold even when supervision is plentiful.
- W2 (head loses object identity through pooled trunk features) means
  it can't distinguish which object it's predicting on.

A **cross-attention head over object tokens** with a contact-context
input simultaneously fixes both:

- Per-body-part learnable query token (5 separate computational paths
  → fixes W1)
- Cross-attention to `(B, 128, d)` object tokens (object identity
  flows directly → fixes W2)
- Frame feature concatenated with `contact_emb` (DAG-correct
  conditioning → enables cross-head info)

So v8 = "affordance attention head" + "DAG-ordered conditioning" +
"consistency loss" is **one architectural change**, not three.

## 3. Architecture spec

### 3.1 Output contract change

| field | v7-fix shape | v8 shape | semantic |
|---|---|---|---|
| `contact_state` | (B, T, 5) sigmoid | (B, T, 5) sigmoid | per-frame per-part contact prob (unchanged) |
| `contact_target_xyz` | (B, T, 5, 3) regression in world frame | (B, T, 5, 3) attention-weighted token xyz (back-compat) | NEW: derived from attention; bounded by object extent |
| `contact_target_attn` | n/a | **(B, T, 5, 128) softmax over object tokens** | NEW primary output: affordance heatmap |
| `phase` | (B, T, 3) softmax | (B, T, 3) softmax | unchanged shape, but now conditional on contact |
| `support` | (B, T, 4) softmax | (B, T, 4) softmax | unchanged shape, conditional on contact + phase |

The `contact_target_xyz` field is preserved as a *derived* quantity for
**back-compat with existing Stage B inference** (which currently
consumes xyz, not attention). It is computed at output time as
`xyz = sum(attn * object_tokens_xyz, dim=-2)` — i.e., the
attention-weighted centroid. This is bounded by the object's
spatial extent (no world-coordinate extrapolation), and on small
objects degenerates to "the predicted token xyz".

Stage B can later migrate to consuming `contact_target_attn` directly
in IntXAttn (v8.5 follow-up). Not in scope for v8.

### 3.2 StructuredHead module

Implementation: `src/piano/models/interaction_predictor.py`. Toggled
via `model.structured_head: true|false` config (default false for
backward compat; v8 config sets it true).

```
StructuredHead:
    inputs: x ∈ (B, T, d), object_tokens ∈ (B, 128, d), object_xyz ∈ (B, 128, 3),
            optional gt_contact ∈ (B, T, 5), optional gt_phase ∈ (B, T)

    # ── Level 0: contact (base) ────────────────────────────────
    contact_logits = MLP_contact(x)                    # (B, T, 5)
    contact_prob   = sigmoid(contact_logits)
    if training and teacher_forcing:
        contact_for_downstream = gt_contact
    else:
        contact_for_downstream = contact_prob
    contact_emb    = Linear(5, d_emb)(contact_for_downstream)     # (B, T, d_emb)

    # ── Level 1a: target (cond on contact) ─────────────────────
    # 5 part queries + frame feature carrying contact context
    x_with_c       = concat([x, contact_emb], dim=-1)  # (B, T, d + d_emb)
    frame_q        = Linear(d + d_emb, d)(x_with_c)    # (B, T, d)
    part_queries   = Param(5, d)                        # learnable
    q              = (frame_q.unsqueeze(2) + part_queries.unsqueeze(0).unsqueeze(0))
                                                        # (B, T, 5, d)
    q_flat         = q.reshape(B, T*5, d)
    # cross-attn outputs attention weights as the affordance heatmap
    _, attn_weights = MultiheadAttention(d, h=6)(
                          q_flat, object_tokens, object_tokens,
                          need_weights=True, average_attn_weights=False)
    # attn_weights: (B, h, T*5, 128) → average over heads → (B, T*5, 128)
    target_attn    = attn_weights.mean(dim=1).reshape(B, T, 5, 128)

    # back-compat xyz output: attention-weighted token positions
    target_xyz     = einsum('btpk,bkc->btpc', target_attn, object_xyz)  # (B, T, 5, 3)

    # ── Level 1b: phase (cond on contact) ──────────────────────
    phase_logits   = MLP_phase(x_with_c)                # (B, T, 3)
    phase_prob     = softmax(phase_logits)
    if training and teacher_forcing:
        phase_for_downstream = gt_phase_one_hot
    else:
        phase_for_downstream = phase_prob
    phase_emb      = Linear(3, d_emb)(phase_for_downstream)

    # ── Level 2: support (cond on contact + phase) ─────────────
    x_full         = concat([x, contact_emb, phase_emb], dim=-1)
    support_logits = MLP_support(x_full)                # (B, T, 4)

    return {
        contact_state, contact_logits,
        contact_target_attn, contact_target_xyz,
        phase, phase_logits,
        support, support_logits,
    }
```

Parameter budget (over independent heads):

- Independent heads: 4 × Linear ≈ 11K params total
- StructuredHead: contact MLP (~150K) + Linear-emb-1 (~0.4K) +
  cross-attn (~590K) + 5 part queries (~2K) + Linear-frame-q (~170K) +
  phase MLP (~150K) + Linear-emb-2 (~0.2K) + support MLP (~150K)
  = **~1.2M params**

Total predictor: 26.1M (v7) → ~27.3M (v8). +5 %, OK.

### 3.3 Loss formulation

Replaces v7-fix's PredictorLoss target_loss term with KL-divergence,
keeps all other heads' losses, adds 4 consistency terms.

#### Target loss (replaces smooth_L1)

GT distribution for `(t, body_part)`, given `contact_target_xyz_gt` =
`p_gt` and 128 object tokens with positions `c_k`:

```
d_k     = ||c_k - p_gt||₂                              # (B, T, 5, 128)
gt_attn = softmax(-d_k² / (2σ²))                       # σ = 0.08 m default
```

σ is the affordance kernel width. Move-as-You-Say uses σ = 0.8 m at
scene scale; we use σ = 0.08 m at object scale (~ 1/10) since our
objects are 10-100× smaller than rooms.

Loss on contact-positive cells only (now well-justified — for
non-contact cells the GT xyz is undefined or far-away and would
inject noise):

```
L_target = KL(gt_attn || pred_attn) on cells with gt_contact > 0.5
         = sum_k gt_attn_k * (log gt_attn_k - log pred_attn_k)
```

Numerically computed via `F.kl_div(log_pred_attn, gt_attn, reduction='none')`
to avoid double log of small numbers.

#### Consistency losses (auxiliary)

```
# 1. Target attention should be peaked when contact, flat otherwise
attn_entropy = -sum_k pred_attn * log pred_attn        # (B, T, 5)
max_entropy  = log(128)
no_contact   = (1 - contact_prob)                      # (B, T, 5)
L_attn_entropy = mean(no_contact * (max_entropy - attn_entropy))

# 2. P(hand_support) ≤ max(P(left_hand_contact), P(right_hand_contact))
hand_contact = max(contact_prob[..., 0], contact_prob[..., 1])
p_hand_supp  = softmax(support_logits)[..., 3]
L_hand_supp_consist = mean(relu(p_hand_supp - hand_contact))

# 3. P(sitting) ≤ P(pelvis_contact)
pelvis_contact = contact_prob[..., 4]
p_sitting      = softmax(support_logits)[..., 2]
L_sit_consist  = mean(relu(p_sitting - pelvis_contact))

# 4. P(phase != non_contact) ≤ P(any_part_contact)
any_contact   = contact_prob.max(dim=-1)
p_in_contact  = 1 - softmax(phase_logits)[..., 0]
L_phase_consist = mean(relu(p_in_contact - any_contact))

L_consistency = (L_attn_entropy + L_hand_supp_consist +
                 L_sit_consist + L_phase_consist) * w_consistency
```

All four are "max-margin" relu hinges that vanish when the constraint
is satisfied. They provide an inductive bias that mirrors the
extraction DAG without forcing exact label match (model may
legitimately disagree with extraction in rare cases).

Default weight: `w_consistency = 0.1`. Sweep candidates: 0.05, 0.2.

### 3.4 Teacher forcing schedule

- Epochs 0-50: `teacher_forcing_prob = 1.0` (always feed GT contact +
  GT phase to downstream heads). Heads learn the conditional structure
  cleanly.
- Epochs 50-80: linear anneal `1.0 → 0.5`. Heads start adapting to
  noisy upstream predictions.
- Epochs 80-100: hold at 0.5.

Standard Bengio et al. NeurIPS 2015 scheduled-sampling pattern.

Per-batch coin flip on `teacher_forcing_prob` (not per-frame). Picking
GT vs prediction is a clip-level decision so the head sees a coherent
upstream signal across the 196 frames.

### 3.5 ObjectEncoder changes

Currently returns only `tokens`. v8 needs token positions for both
GT attention computation and the `target_xyz` back-compat output.

```python
def forward(self, point_cloud):
    xyz = point_cloud
    feat = None
    xyz, feat = self.sa1(xyz, feat)
    xyz, feat = self.sa2(xyz, feat)
    feat = self.refine(feat)
    return xyz, feat        # was: return feat
```

The (xyz, features) pair flows through the predictor; the attention
head consumes both.

Random-init FPS in `_fps()` produces non-deterministic token
selection per forward pass. **OK for v8** because the GT attention is
recomputed per batch using that batch's actual token xyz. Tokens are
spatially well-distributed regardless of init seed (FPS coverage
property), so the attention output is well-defined.

### 3.6 Eval metric changes

Old: target overall L2, per-part L2, <5cm/<10cm/<20cm hit.

New (in `scripts/stage_a_predictor/eval_predictor.py`):

- **target_top1_token_recall**: of contact-positive (frame, part)
  cells, fraction where argmax(pred_attn) == argmax(gt_attn)
- **target_top3_token_recall**: ... where argmax(gt_attn) ∈ top-3 of
  pred_attn
- **target_top5_token_recall**: ... where argmax(gt_attn) ∈ top-5
- **target_attn_kl**: mean KL(gt_attn || pred_attn) on contact cells
- **target_attn_emd_proxy**: distance from argmax(pred_attn)'s xyz to
  argmax(gt_attn)'s xyz, in cm — Earth-Mover-Distance proxy
- **target_xyz_l2_legacy**: existing L2 metric on the
  attention-weighted xyz (so we can compare to v7-fix on the same
  metric)

Contact / phase / support metrics unchanged.

Acceptance for v8 launch (vs v7-fix):

- target_top1_token_recall ≥ 0.30
- target_xyz_l2_legacy ≤ 18 cm (≥ 3 cm improvement vs 21.77)
- contact macro_f1 ≥ 0.24 (≥ v7-fix's 0.237)
- phase macro F1 ≥ 0.62 (within noise of v7-fix's 0.632)
- support macro F1 ≥ 0.40 (≥ v7-fix's 0.397)

### 3.7 Stage B integration plan (out of scope for v8 prototype, but documented)

Two-step migration:

1. **v8 (this round)**: Stage B unchanged. Consumes `contact_target_xyz`
   from predictor (now attention-derived xyz). Should already see a
   bump because the xyz output is more accurate.
2. **v8.5 (after v8 trained)**: Replace IntXAttn's xyz-based
   conditioning with attention-based. The conditioning becomes
   `attn @ object_features` (what the predictor *thinks* should be
   attended to in object features), instead of `MLP(xyz)`. Closer to
   how Move-as-You-Say's AMDM consumes affordance.

## 4. Implementation file plan

| file | change |
|---|---|
| `src/piano/models/object_encoder.py` | `forward` returns `(xyz, features)` instead of `features`. Add `feature_dim` property. ~5 lines. |
| `src/piano/models/interaction_predictor.py` | New `StructuredHead` class. Extend `InteractionPredictor.__init__` with `structured_head` flag. Extend `forward` to dispatch and accept `gt_contact`, `gt_phase`, `teacher_forcing`. Pass `(xyz, features)` to head. ~120 lines. |
| `src/piano/training/losses.py` | Extend `PredictorLoss` with `target_loss_kind: "smooth_l1" \| "kl_div"`. Add `_kl_div_target_loss(pred_attn, gt_xyz, object_xyz, sigma)` helper. Add `_consistency_loss(out, contact_prob, mask)`. ~80 lines. |
| `src/piano/training/train_predictor.py` | Pass `gt_contact`, `gt_phase` into model. Compute `teacher_forcing_prob` from epoch + schedule. Pass `object_xyz` from encoder to loss. ~30 lines. |
| `configs/model/interaction_predictor.yaml` | Add `structured_head`, `target_loss_kind`, `target_kernel_sigma`, `consistency_weight`, `teacher_forcing_*` flags. |
| `configs/training/predictor_v8_structured.yaml` | New config with all v8 flags on. |
| `scripts/stage_a_predictor/eval_predictor.py` | New token-level metrics. Keep legacy L2 for back-compat. ~60 lines. |
| `tests/test_structured_head.py` | New: forward shape, loss compute, backward, teacher forcing toggle, consistency loss zero on perfect input. ~80 lines. |

Total: ~440 lines of new + edited code.

## 5. Sanity tests (must pass before pushing)

Local tests in `tests/test_structured_head.py`:

1. **Forward shape**: `StructuredHead(...)(x, tokens, xyz)` returns
   correct shapes for each output field.
2. **Backward**: `loss.backward()` populates `.grad` on every parameter
   of trunk + head + ObjectEncoder.
3. **Teacher forcing toggle**: with `teacher_forcing=True`, downstream
   heads see GT; with `False`, they see model predictions. Verify by
   monkey-patching contact head to return constant 0 and checking that
   downstream behaviour matches expectation under each mode.
4. **KL loss positivity**: `_kl_div_target_loss(pred, gt, ...)` ≥ 0,
   = 0 when pred = gt.
5. **Consistency loss**: returns 0 when constraints exactly satisfied
   (e.g., support_prob = 0 everywhere, contact_prob = 1 everywhere
   should yield zero relu hinges).
6. **Back-compat**: `structured_head: false` config produces identical
   output to the v7-fix predictor (regression test).

## 6. Risk register

| risk | likelihood | mitigation |
|---|---|---|
| KL divergence numerical instability (log of small probs) | medium | use `F.kl_div(log_pred, gt, reduction='none')` form; clamp gt to [1e-8, 1] |
| Non-deterministic FPS hurts training stability | low | each forward gets fresh GT attention against that batch's tokens; FPS coverage is stable in distribution even if order isn't; if issue arises, switch FPS init to deterministic per-object |
| Teacher forcing → train-test gap | medium | scheduled sampling (anneal to 0.5); sanity test that without TF the model still trains |
| Consistency loss dominates and prevents head specialisation | low | weight = 0.1 (small); sweep 0.05 / 0.1 / 0.2 |
| Per-part queries collapse to one mode (5 queries become identical) | low | initialise with N(0, 0.02); add per-query positional bias if needed |
| target xyz back-compat output is no longer differentiable wrt object | n/a | attention-weighted xyz IS differentiable (sum of soft weights × xyz) |
| Stage B's existing inference code breaks | low | back-compat output is `target_xyz` shape (B, T, 5, 3); same contract; quality of xyz changes but interface stays identical |

## 7. Acceptance gate for shipping v8

Run server training (~6 h) with `predictor_v8_structured.yaml`, eval
on val 1304 clips. Pass criteria (Section 3.6):

- target_top1_token_recall ≥ 0.30
- target_xyz_l2_legacy ≤ 18 cm
- contact macro_f1 ≥ 0.24
- phase macro F1 ≥ 0.62
- support macro F1 ≥ 0.40

If any of (contact / phase / support) regresses below v7-fix:
- Sweep `consistency_weight` in {0, 0.05, 0.1, 0.2}
- Sweep `teacher_forcing` schedule

If `target_top1_token_recall < 0.30` despite head changes:
- Sweep `target_kernel_sigma` in {0.05, 0.08, 0.12, 0.20}
- Add Point Transformer pretraining (W6, deferred) as v9

## 8. References

- Wang, Z., Chen, Y., Jia, B., Li, P., et al. "Move as You Say,
  Interact as You Can: Language-guided Human Motion Generation with
  Scene Affordance." CVPR 2024 (Highlight). arXiv: 2403.18036.
- Xu, D., Ouyang, W., Wang, X., Sebe, N. "PAD-Net: Multi-tasks Guided
  Prediction-and-Distillation Network for Simultaneous Depth
  Estimation and Scene Parsing." CVPR 2018.
- Misra, I., Shrivastava, A., Gupta, A., Hebert, M. "Cross-stitch
  Networks for Multi-task Learning." CVPR 2016.
- Vandenhende, S., Georgoulis, S., Van Gool, L. "MTI-Net: Multi-Scale
  Task Interaction Networks for Multi-Task Learning." ECCV 2020.
- Liao, Y., et al. "PPDM: Parallel Point Detection and Matching for
  Real-time Human-Object Interaction Detection." CVPR 2020.
  arXiv: 1912.12898.
- Pavlakos, G., et al. "Expressive Body Capture: 3D Hands, Face, and
  Body from a Single Image." CVPR 2019. arXiv: 1904.05866.
- Bengio, S., Vinyals, O., Jaitly, N., Shazeer, N. "Scheduled Sampling
  for Sequence Prediction with Recurrent Neural Networks." NeurIPS
  2015. arXiv: 1506.03099.
- Carion, N., et al. "End-to-End Object Detection with Transformers."
  ECCV 2020 (DETR). arXiv: 2005.12872.

Companion docs:
- `analyses/2026-05-05_v7fix_results_and_v6_baseline_correction.md`
- `analyses/2026-05-01_v17_diagnostics_and_gumbel.md` (γ_int evidence)
- `analyses/2026-05-03_gamma_int_re_evaluation.md`
- `analyses/2026-05-03_pseudo_label_v12_strict_design.md` (v12 labels)
