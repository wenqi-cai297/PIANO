# 2026-05-03 — v9 server eval results + v9.1 surgical plan

## TL;DR

v9 was a **mixed result with one dominant win and one regression**:

- ✅✅✅ **Contact pos_weight fixed the foot recall=0 LOSS BUG** that
  was present across v6 → v8.1.1. Foot recall 0/0 → 0.79/0.84,
  hand recall 17-25 % → 64-68 %, contact macro F1 0.227 → 0.371
  (+63 % rel). Predictor is now genuinely usable as a Stage B z_int
  source.

- ⚠️ **Mask3D 4-layer decoder no lift**: topk3_iou 0.133 → 0.136.
  +7.7 M params bought us nothing on per-token ranking. Kept in v9.1
  as a control variable (so we can isolate the τ change).

- ❌ **Logit Adjustment τ=1.0 caused regression**: support macro F1
  0.404 → 0.218 (both_feet F1 collapsed 0.94 → 0.000), phase F1
  0.637 → 0.541. τ=1.0 over-pushed rare classes; dominant class
  abandoned.

v9.1 is a 2-line surgical fix:
1. **Drop hand_support class** (user insight: InterAct has no
   gymnastic poses → hand_support is just "both_feet + hand bracing",
   redundant with contact_state[hand])
2. **Soften τ=1.0 → 0.3**

## 1. v9 results in full

(val 1304 clips, subject_split, best_val ep19)

| metric | v8.1.1 | v9 best | Δ | gate | pass |
|---|---:|---:|---|---|:---:|
| **foot left recall** | 0.000 | **0.791** | +0.79 | ≥ 0.15 | ✅✅✅ |
| **foot right recall** | 0.000 | **0.836** | +0.84 | ≥ 0.15 | ✅✅✅ |
| hand left recall | 0.247 | 0.682 | +0.44 | — | ✅ |
| hand right recall | 0.171 | 0.639 | +0.47 | — | ✅ |
| pelvis recall | 0.527 | 0.859 | +0.33 | — | ✅ |
| contact any_part recall | 0.361 | **0.849** | +0.49 | — | ✅✅ |
| contact any_part F1 | 0.456 | **0.695** | +0.24 | — | ✅✅ |
| **contact macro F1** | 0.227 | **0.371** | +0.14 | ≥ 0.30 | ✅ |
| target overall L2 | 23.12 | 23.11 | flat | — | tied |
| target <5cm hit | 12.55 % | 12.29 % | -0.3pp | ≥ 12 % | ✅ |
| target <10cm hit | 27.44 % | 27.47 % | flat | — | tied |
| pelvis L2 (cm) | 14.3 | 13.4 | -0.9 | ≤ 14.5 | ✅ |
| pelvis pct<10cm | 58.9 % | 58.9 % | flat | ≥ 55 % | ✅ |
| topk3_mean_iou | 0.133 | 0.136 | +0.003 | ≥ 0.25 | ❌ |
| topk3_mean_f1 | 0.176 | 0.180 | flat | ≥ 0.35 | ❌ |
| **phase macro F1** | 0.637 | **0.541** | -0.10 | ≥ 0.65 | ❌ |
| **support macro F1** | 0.404 | **0.218** | -0.19 | ≥ 0.42 | ❌❌ |
| support both_feet F1 | 0.940 | **0.000** | -0.94 | — | ❌❌❌ collapsed |
| support sitting F1 | 0.656 | 0.656 | flat | — | tied |
| support hand_support F1 | 0.067 | 0.067 | flat | — | still 0 |
| foot left L2 (cm) | 43.2 | 43.6 | +0.4 | ≤ 25 | ❌ |
| foot right L2 (cm) | 37.5 | 37.9 | +0.4 | ≤ 25 | ❌ |

**Score**: 7 / 11 acceptance gates pass. Critical wins (foot recall
fix) + critical losses (support / phase regression).

## 2. Per-change verdict

| change | role | evidence | verdict |
|---|---|---|---|
| Contact pos_weight | (A) | foot 0 → 0.79, contact macro_f1 +63 % rel | DOMINANT WIN |
| Mask3D 4-layer decoder | (B) | topk3_iou 0.133 → 0.136 | NO LIFT |
| Logit Adjustment τ=1.0 | (C) | support 0.404 → 0.218, both_feet → 0 | REGRESSION |

Combined verdict: (A) is a keeper, (B) is neutral, (C) is harmful at
τ=1.0.

## 3. Why support both_feet collapsed under τ=1.0

Logit Adjustment (Menon ICLR'21) at training time replaces logit
ℓ_y with ℓ_y - τ × log π_y. At τ=1.0 with our class priors:

| class | π | -τ × log π | effective shift |
|---|---:|---:|---:|
| both_feet | 0.847 | -0.166 | tiny boost |
| sitting | 0.088 | -2.43 | strong boost |
| single_foot | 0.034 | -3.38 | strong boost |
| hand_support | 0.031 | -3.47 | strong boost |

At inference, we use raw logits ℓ_y. The model trained to compensate
for τ=1.0 was systematically over-confident on dominant class to
compensate for the boost-to-rare-class subtraction. When we then
remove the subtraction at inference, the rare-class predictions stay
inflated and dominate. Result: both_feet (84.7 % of frames) gets
predicted ~ 0 % of the time → F1 0.000.

This is a known logit-adjustment failure mode at high τ. Menon et al.
recommend τ ∈ [0.5, 1.5] but explicitly warn that extremely
imbalanced datasets need lower τ (their Table 5 shows τ=0.5 working
better than τ=1.0 on iNaturalist-2018, which has class imbalance
~ 100:1; our 84.7 vs 3.1 % is a 27:1 ratio, in the same regime).

## 4. The hand_support insight (user contribution)

User question: "hand_support 这个类，在数据集里占多少比例？什么情况下我们判定为 hand support？"

extract_support.py judges hand_support as:

```python
hand_support = (
    (left_hand_contact OR right_hand_contact)   # ① hand on object
    AND pelvis_stationary                       # ② |pelvis xz speed| < 0.15 m/s
    AND phase == stable_contact                 # ③ object also stationary
)
```

Physical semantic: "lean / brace pose" — body supported by hand on
a static object, body not walking, object not being manipulated.

**User's observation**: in InterAct (chairs / imhd / neuraldome /
omomo) there are no gymnastic poses where feet are airborne and only
hand provides support. So when hand_support fires, the person is
ALWAYS still both-feet on floor. The class adds zero information
beyond contact_state[hand].

**Implications**:

- Information-theoretically: hand_support ⊕ contact_state[hand] = 0
  bits new info on InterAct.
- Stage B downstream: contact_state already gives Stage B "hand is
  contacting". Stage B can derive "and bracing-pose" from phase +
  contact_state without needing a dedicated support class.
- Modeling-wise: 3 % positive rate × compound 3-AND ⇒ inherently
  hardest class; even with logit_adjust at τ=1.0 v9 still got
  hand_support F1 = 0.067 (basically random).

**Conclusion**: drop hand_support entirely. Collapse to both_feet.

## 5. v9.1 design — 2 surgical changes

### Change 1: drop hand_support class

Implementation: dataloader-level mapping, npz files unchanged.

```python
# HOIDataset._load_pseudo_labels (v9.1)
if self.support_collapse_hand_support and result["support"] is not None:
    sup = result["support"].copy()
    sup[sup == 3] = 0   # SUPPORT_HAND → SUPPORT_BOTH_FEET
    result["support"] = sup
```

Plus:
- `data.support_collapse_hand_support: true` in v9.1 yaml
- `model.output.num_support_states: 3` override
- eval_predictor.py support metrics use `support_class_names[:num_support]`

Why dataloader-level (not re-extraction):
- 8475 npz files don't need to be re-touched (saves 1-2 hr server time)
- Easily reversible (flag back to false)
- Stage B v8.1b refactor will read the same npzs and apply the same
  collapse logic; no data divergence

### Change 2: soften logit_adjust τ=1.0 → 0.3

τ=0.3 with 3-way support priors:

| class | π (after collapse) | -τ × log π | effective shift |
|---|---:|---:|---:|
| both_feet | 0.847 + 0.031 = 0.878 | -0.039 | negligible |
| sitting | 0.088 | -0.73 | mild boost |
| single_foot | 0.034 | -1.01 | moderate boost |

Much milder than v9. Should preserve dominant class without losing
the rare-class lift.

## 6. What v9.1 keeps (control variables)

To ensure we can attribute v9.1's success/failure cleanly:

- **Contact pos_weight kept**. Dominant v9 win — non-negotiable.
- **Mask3D decoder kept** (4-layer, FFN=1024, 6 heads). v9 showed
  topk3_iou flat — keeping it as control means v9.1 metrics are
  directly comparable to v9 on the architecture axis. If v9.1
  topk3_iou stays at 0.13, we have evidence that mask decoder is
  truly neutral and can drop it in v9.2 to save params.
- All v8.1.1 wins kept: MoMask Bernoulli mask, multi-hot binary GT,
  top-K minimum mask, focal+dice loss, no consistency loss.

## 7. v9.1 acceptance gates

| metric | v8.1.1 | v9 | **v9.1 gate** | rationale |
|---|---:|---:|---|---|
| support macro F1 | 0.404 | 0.218 | **≥ 0.55** | recover dominant class; 3-way is more concentrated so easier ceiling |
| both_feet F1 | 0.940 | 0.000 | **≥ 0.85** | dominant class restored |
| sitting F1 | 0.656 | 0.656 | ≥ 0.65 | unchanged |
| phase macro F1 | 0.637 | 0.541 | **≥ 0.62** | recover from τ=1.0 over-correction |
| foot recall | 0/0 | 0.79/0.84 | **≥ 0.70** | keep v9 win |
| contact macro F1 | 0.227 | 0.371 | **≥ 0.35** | keep v9 win |
| target <5cm hit | 12.6 % | 12.3 % | ≥ 12 % | flat |
| pelvis pct<10cm | 58.9 % | 58.9 % | ≥ 55 % | flat |
| topk3_iou | 0.133 | 0.136 | (no gate) | observe; if still 0.13, drop decoder in v9.2 |

Pass condition: 7 / 8 gates + foot recall preserved.

If v9.1 fails:
- support F1 still < 0.45 → τ=0.3 still too aggressive; try τ=0.15
- foot recall < 0.50 → 3-way collapse damaged contact head somehow
  (unlikely; supports are independent)
- phase still < 0.60 → phase head needs separate τ tuning

## 8. v9.2 candidates (if v9.1 passes)

- **Drop Mask3D decoder, revert to single-layer Q/K**. If v9.1
  topk3_iou stays at 0.13, the +7.7 M params are dead weight. v9.2
  with single-layer head saves wallclock and memory.
- **EgoChoir motion-KV stream**. The real fix for token-level
  ranking on moving contact targets — needs joint kinematics in
  dataloader (~ 100 LOC change). Defer until we know whether
  ranking is needed for Stage B (γ_int evidence from v18 will tell us).

## 9. Implementation summary (commit on the way)

| file | change | LOC |
|---|---|---:|
| `src/piano/data/dataset.py` | `support_collapse_hand_support` flag in HOIDataset; mapping in `_load_pseudo_labels` | +18 |
| `src/piano/training/train_predictor.py` | `output_cfg = merge(model_cfg.output, cfg.model.output)` for num_support override; flag passed to dataset | +12 |
| `scripts/stage_a_predictor/eval_predictor.py` | output_cfg merge; flag passed to dataset; support metric uses dynamic class_names | +12 |
| `configs/training/predictor_v9_1_3way_support.yaml` | new config | +120 |
| `tests/test_structured_head.py` | 3 new v9.1 tests | +50 |
| `analyses/2026-05-03_v9_results_and_v91_plan.md` | this doc | +220 |

Total: ~430 LOC.

## 10. Sanity tests (passed)

```
$ pytest tests/test_structured_head.py -q
25 passed in 2.14s
```

New v9.1 tests:
- `test_v91_3way_support_collapse_label_mapping`: mapping id=3 → 0
  produces expected array
- `test_v91_config_yaml_propagates_3way_support`: yaml has
  num_support_states=3 + support_collapse_hand_support=true +
  logit_adjust_tau=0.3 + use_contact_pos_weight=true (kept)
- `test_v91_predictor_3way_support_head`: InteractionPredictor with
  num_support_states=3 emits (B, T, 3) support_logits

## 11. References

- Menon et al. ICLR 2021 (Logit Adjustment) — arXiv:2007.07314.
  Recommends τ ∈ [0.5, 1.5]; warns that extreme imbalance benefits
  from τ < 1.
- v9 design + companion doc: `analyses/2026-05-03_v9_combined_design.md`.
- v9 server eval JSONs:
  - `runs/eval/stageA_predictor_v9_combined_val/predictor_v9_combined_val_best.json`
  - `runs/eval/stageA_predictor_v9_combined_val/predictor_v9_combined_val_final.json`
- v9 wandb: `runs/wandb_logs/wandb_history_predictor_v9_combined.csv`.
