# 2026-05-03 — Stage A v9: combined loss-bug fix + Mask3D-style decoder

## TL;DR

Three changes in **one** retrain, each with a specific failure mode it
addresses, each grounded in 2024-2026 SOTA:

| change | failure addressed | citation | expected lift |
|---|---|---|---|
| (A) class-balanced contact BCE (per-part `pos_weight`) | foot recall = 0; hand recall 17-25 % | DECO ICCV'23, HACO NeurIPS'25 | foot 0 → 0.20-0.40, hand → 0.35-0.50 |
| (B) Mask3D-style multi-layer mask decoder | topk3_iou = 0.13 plateau on hand/foot | InteractVLM CVPR'25 (DAMON F1 55 → 76, +20pp) | topk3_iou 0.13 → 0.30+ |
| (C) Logit Adjustment for phase + support | already coded but disabled in every yaml | Menon ICLR'21 | phase / support +2-4 pp free |

**Path B** still in effect: predictor outputs `contact_target_attn`
(per-token sigmoid mask). v18 will consume it after Stage B v8.1b
refactor.

22/22 sanity tests pass. v9 predictor is 34.7 M params (v8.1 was
26.9 M, +28.8 %). Trunk + ObjectEncoder unchanged.

## 1. Failure-mode-driven design

v8.1.1 server eval (best_val ep 29) revealed three independent failure
modes after we'd already moved past the v7-fix architectural floor:

| failure | v7-fix | v8 | v8.1 | v8.1.1 | gate | dominant cause |
|---|---:|---:|---:|---:|---|---|
| foot recall (left/right) | 0/0 | 0/0 | 0/0 | 0/0 | ≥ 0.15 | **loss bug** (no pos_weight) |
| hand_support recall | 0 | 0 | 0 | 0 | ≥ 0.15 | structural compound class |
| topk3_iou | n/a | n/a | 0.120 | 0.133 | ≥ 0.30 | **head capacity** |
| target <5cm hit | 4.5 % | 5.6 % | 11.6 % | 12.6 % | ≥ 11 % | passing |
| phase macro F1 | 0.632 | 0.577 | 0.637 | 0.637 | ≥ 0.62 | passing |

The **loss bug** discovery is the most important: contact head uses
bare `F.binary_cross_entropy_with_logits` with no `pos_weight` and no
focal. The `focal_gamma=2.0` flag in every prior yaml only applies to
phase + support cross-entropy. With foot positive rate ~3 %, BCE is
dominated 32:1 by negatives → model trivially predicts "negative
everywhere" → recall = 0. This is not a data sparsity problem; it's a
loss configuration problem.

## 2. Change A — class-balanced contact BCE

### Code: `src/piano/training/losses.py:285`

Before (every prior version):

```python
loss_contact = F.binary_cross_entropy_with_logits(
    pred["contact_logits"], gt_contact, reduction="none",
)  # (B, T, 5)
```

After:

```python
contact_pw = getattr(self, "contact_pos_weight", None)
if contact_pw is not None:
    pw = contact_pw.view(1, 1, -1).to(...)
    loss_contact = F.binary_cross_entropy_with_logits(
        pred["contact_logits"], gt_contact,
        reduction="none", pos_weight=pw,
    )
else:
    loss_contact = F.binary_cross_entropy_with_logits(...)  # back-compat
```

`contact_pos_weight` is a `(num_body_parts,)` registered buffer
computed at training start.

### Computation: `src/piano/training/train_predictor.py:498`

Extends `compute_class_priors` to also return per-part contact
positive rates, then:

```python
pw = ((1 - π_part) / π_part).clamp(max=15)
```

For v12_strict label set on InterAct (~50% any-part contact frac):
- π_pelvis ≈ 0.30 → pw ≈ 2.3
- π_left_hand ≈ 0.20 → pw ≈ 4.0
- π_right_hand ≈ 0.18 → pw ≈ 4.6
- π_left_foot ≈ 0.018 → pw ≈ 15.0 (capped)
- π_right_foot ≈ 0.024 → pw ≈ 15.0 (capped)

The cap = 15 prevents runaway weights when a body part is essentially
absent from the data (avoids pw = 100+ on extraction failures).

### Literature support

- **DECO** (Tripathi et al., ICCV 2023, [github 250★](https://github.com/sha2nkt/deco))
  uses class-balanced BCE for per-vertex body-scene contact, explicitly
  rejecting focal loss as ineffective in the "passive zero" regime.

- **HACO** (Jung & Lee, NeurIPS 2025, arXiv:2505.11152,
  [github 54★](https://github.com/dqj5182/HACO_RELEASE)) reports per-element
  contact F1 0.522 with class-balanced BCE vs 0.409 with focal — same
  problem class as ours.

- **Asymmetric Loss** (Ben-Baruch et al., ICCV 2021, arXiv:2009.14119,
  [github 797★](https://github.com/Alibaba-MIIL/ASL)) is the explicit
  fix for "passive zero" pathology where focal `(1-p_t)^γ` modulator
  saturates before the model is incentivised to flip its prediction.
  pos_weight is the simpler primitive, ASL is the upgrade if pos_weight
  alone is not enough.

- **Class-Balanced Loss** (Cui et al., CVPR 2019, arXiv:1901.05555)
  formalises `(1 - β^n_y) / (1 - β)` with β = 0.999 — equivalent to
  pos_weight at our regime when n_y is the per-class positive count.

## 3. Change B — Mask3D-style multi-layer mask decoder

### Architectural detail

Replace v8.1.1's `CrossAttentionWeightsOnly` (single-layer Q/K-only
attention) with `AffordanceMaskDecoder`:

```
Input:  frame_q ∈ (B, T, d=384)  obj_tokens ∈ (B, M=128, d=384)

Build (B, T, P=5, d) queries:
  q[b, t, p] = frame_q[b, t] + part_query[p]

Decoder (4 layers):
  for l in range(4):
    q = q + self_attn(q)        # queries refine across (frame, part) pairs
    q = q + cross_attn(q, obj)  # attend to object features
    q = q + FFN(q)              # GELU(Linear(d→1024)→Linear(1024→d))

Mask projection:
  q_proj = Linear(d → d/2)(q)        # (B, T*P, mask_dim)
  k_proj = Linear(d → d/2)(obj)      # (B, M, mask_dim)
  logits = (q_proj @ k_proj^T) * (mask_dim ** -0.5)   # (B, T*P, M)

Output: logits.reshape(B, T, P, M)  # raw, caller does sigmoid
```

Param budget:
- v8.1.1 single-layer head: ~ 0.6 M params (q_proj, k_proj only)
- v9 4-layer mask decoder: ~ 8.3 M params
- net delta on full predictor: 26.9 M → 34.7 M (+ 28.8 %)

### Why this is the right architectural change

- **Per-frame query refinement**. Each (frame, part) query gets 4
  rounds of refinement: self-attn lets it talk to neighbouring
  (frame', part') queries (temporal + cross-part coordination); cross-
  attn binds it to the right object token cluster.
  
- **Frozen V/out_proj problem solved**. Q/K-only attention (v8.1.1)
  avoided DDP unused-param crashes but had only one Q × K^T pass with
  no FFN refinement. The decoder uses full attention internally and
  then collapses to Q · K^T at the END, so V/out_proj are used INSIDE
  the decoder layers and the final mask projection is the sole
  unprojected step.

- **InteractVLM is direct precedent**. Same task class (per-part
  contact prediction on body-mesh tokens), same input modality
  (transformer trunk + cross-attn over PC-derived tokens), and they
  show **+20 pp absolute on DAMON F1** by switching from DECO's
  body-part attention to a SAM-style mask decoder. Our v8.1 head is
  closer to DECO than to InteractVLM; this change closes the gap.

### Literature support

- **InteractVLM** (Dwivedi et al., CVPR 2025, arXiv:2504.05303) —
  SAM mask decoder + focal+L1 → DAMON F1 55.0 → 75.6.
- **Mask3D** (Schult et al., ICRA 2023, arXiv:2210.03105,
  [github 700+ ★](https://github.com/JonasSchult/Mask3D)) — 6-layer
  query decoder, the architectural reference for query-based dense
  prediction over PC-derived tokens.
- **OneFormer3D** (Kolodiazhnyi et al., CVPR 2024, arXiv:2311.14405,
  [github 200+ ★](https://github.com/oneformer3d/oneformer3d)) —
  multi-task generalization of Mask3D.

### Encoder upgrade explicitly NOT done

We considered upgrading PointNet++ → PointNeXt / PT V3 / Sonata.
**Heuken et al. (arXiv:2504.18355)** directly ablates PC backbones on
3D AffordanceNet:

| Backbone | mIoU |
|---|---:|
| PointNet++ | **21.5** |
| DGCNN | 18.4 |
| PT V3 | 12.8 |

PT V3 *loses* 8.7 mIoU to PointNet++ on dense per-point affordance.
Sonata (CVPR 2025) is scene-scale, not object-scale; Uni3D (ICLR 2024)
is whole-object semantic categories, not dense affordance. **Don't
burn a retrain on encoder upgrade.**

## 4. Change C — enable Logit Adjustment

`losses.py:236` already implements logit adjustment as registered
buffers, but every prior yaml had `use_logit_adjustment: false`. v9
yaml flips it on:

```yaml
use_logit_adjustment: true
logit_adjust_tau: 1.0
```

Reference: **Menon et al., ICLR 2021** (arXiv:2007.07314) — at training
time, add τ × log π_y to logits before cross-entropy; at inference
use raw logits. Net effect: shifts the decision boundary toward rare
classes without changing the model. Free 2-4 pp lift on phase and
support macro F1 expected.

## 5. What we DON'T change

| component | status | reason |
|---|---|---|
| Trunk (10 layers, d=384, h=6, ffn=1024) | unchanged | DiT scaling laws say 10K clips are 4 orders of magnitude below saturation; v8.1 phase F1 0.637 proves trunk learns simple heads fast |
| Object encoder (PointNet++ 1024→512→128) | unchanged | Heuken 2025 ablation shows PointNet++ > PT V3 by 8.7 mIoU on dense affordance |
| MoMask Bernoulli mask conditioning | kept | v8.1 win — phase F1 fixed |
| Multi-hot binary GT + top-K=3 union | kept | v8.1.1 win — covers foot empty-mask issue |
| focal+dice on multi-hot mask | kept | v8.1 win — target <5cm hit tripled |
| Drop consistency loss | kept | v8.1 evidence — was being ignored |
| Path B (no xyz back-compat) | kept | aligns with v8.1b Stage B refactor |

## 6. Acceptance gates

| metric | v7-fix | v8 | v8.1 | v8.1.1 | **v9 gate** |
|---|---:|---:|---:|---:|---|
| foot left recall | 0 | 0 | 0 | 0 | **≥ 0.15** (FIX) |
| foot right recall | 0 | 0 | 0 | 0 | **≥ 0.15** (FIX) |
| topk3_mean_iou | n/a | n/a | 0.120 | 0.133 | **≥ 0.25** |
| topk3_mean_f1 | n/a | n/a | 0.162 | 0.176 | ≥ 0.35 |
| target <5cm hit | 4.5 % | 5.6 % | 11.6 % | 12.6 % | ≥ 12 % (no regress) |
| contact macro_f1 | 0.237 | 0.235 | 0.219 | 0.227 | **≥ 0.30** (foot fix lifts macro) |
| phase macro F1 | 0.632 | 0.577 | 0.637 | 0.637 | ≥ 0.65 (logit_adjust lift) |
| support macro F1 | 0.397 | 0.378 | 0.393 | 0.404 | ≥ 0.42 (logit_adjust lift) |
| pelvis L2 (cm) | 15.4 | 14.4 | 14.8 | 14.3 | ≤ 14.5 (no regress) |
| pelvis pct<10cm | 33.9 % | 51.9 % | 59.1 % | 58.9 % | ≥ 55 % |
| foot L2 (cm) | 25.3 | 27.4 | 42.7 | 40.4 | **≤ 25** (FIX) |

Pass condition: 8/11 gates + the 3 FIX gates (foot recall × 2 + foot L2).

If v9 still fails:
- foot recall < 0.10 → upgrade to ASL loss (v9.1, ~ 30 LOC, ~1 hour)
- topk3_iou < 0.20 → add EgoChoir motion-KV stream (v9.2, ~ 100 LOC,
  needs joint kinematics in dataloader)
- contact macro_f1 < 0.25 → check if pos_weight cap is hurting hand
  precision; tune cap = 8 instead of 15

## 7. File plan

| file | change | LOC |
|---|---|---:|
| `src/piano/models/interaction_predictor.py` | `AffordanceMaskDecoder` class; `target_attn_kind` flag | +130 |
| `src/piano/training/losses.py` | `contact_pos_weight` param in `PredictorLoss`; apply via `pos_weight=` | +30 |
| `src/piano/data/dataset.py` | extend `compute_class_priors` to return contact_part_freq | +20 |
| `src/piano/training/train_predictor.py` | wire `use_contact_pos_weight` flag, compute and pass `pos_weight` | +30 |
| `scripts/stage_a_predictor/eval_predictor.py` | propagate v9 flags through `_build_models` | +10 |
| `configs/training/predictor_v9_combined.yaml` | new | +160 |
| `tests/test_structured_head.py` | 4 new v9 tests | +160 |

Total ~540 LOC.

## 8. Sanity test status

```
$ pytest tests/test_structured_head.py -q
22 passed in 1.91s
```

New v9 tests:
- `test_v9_mask_decoder_forward_shape`: AffordanceMaskDecoder produces
  (B, T, P, M) logits; every parameter receives gradient at backward
  (DDP regression).
- `test_v9_structured_head_with_mask_decoder`: full StructuredHead with
  mask_decoder + mask conditioning + logits output runs forward +
  backward; **caught a real DDP bug** where StructuredHead's redundant
  `part_queries` Parameter was unused under mask_decoder kind.
- `test_v9_contact_pos_weight_increases_positive_loss`: pos_weight=32
  on foot-positive cells produces 13× more contact loss than the
  unweighted version (0.69 vs 9.29).
- `test_v9_config_yaml_end_to_end`: `predictor_v9_combined.yaml`
  builds correctly with all v9 flags propagated.

## 9. Server retrain budget

- 4-layer mask decoder vs single-layer: ~ +15 % wallclock (decoder
  forward + backward through 4 self-attn + cross-attn + FFN sublayers
  on 980 query × 128 key shapes). v8.1.1 was ~6 h → v9 estimated ~7 h.
- Memory: T*P*d = 196*5*384 = 376 K query elements; 4-layer self-attn
  is 4 × 980² ≈ 3.8 M attention scores per batch. Manageable on A6000.
- Class prior scan adds ~30 s startup overhead.

## 10. References

- Tripathi et al. ICCV 2023 (DECO) — arXiv:2309.15273. [github](https://github.com/sha2nkt/deco)
- Jung & Lee NeurIPS 2025 (HACO) — arXiv:2505.11152. [github](https://github.com/dqj5182/HACO_RELEASE)
- Ben-Baruch et al. ICCV 2021 (ASL) — arXiv:2009.14119. [github](https://github.com/Alibaba-MIIL/ASL)
- Cui et al. CVPR 2019 (Class-Balanced Loss) — arXiv:1901.05555.
- Menon et al. ICLR 2021 (Logit Adjustment) — arXiv:2007.07314.
- Dwivedi et al. CVPR 2025 (InteractVLM) — arXiv:2504.05303.
- Schult et al. ICRA 2023 (Mask3D) — arXiv:2210.03105. [github](https://github.com/JonasSchult/Mask3D)
- Kolodiazhnyi et al. CVPR 2024 (OneFormer3D) — arXiv:2311.14405. [github](https://github.com/oneformer3d/oneformer3d)
- Heuken et al. arXiv:2504.18355 — PointNet++ vs PT V3 ablation on dense
  affordance.

Companion analyses:
- `analyses/2026-05-02_class_imbalance_sota_survey.md`
- `analyses/2026-05-02_predictor_v9_architecture_research.md`
- `analyses/2026-05-02_hoi_data_aug_synthetic_transfer_survey.md`
- `analyses/2026-05-05_v8_round1_diagnosis_and_v81_plan.md`
- `analyses/2026-05-05_v81_results_and_v811_plan.md`
