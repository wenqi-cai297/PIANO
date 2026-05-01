# 2026-05-04 — Stage A v7 (v12-strict labels) target-head 21 cm L2 disaster + v7-fix

User feedback after Stage A v7 eval:

> "我们 pseudo label 都重新提取了，不可能再用旧的 predictor 了，
> 这 pipeline 上就说不过去。必须把 predictor 重新修好。"

This doc captures the v7 failure diagnosis, root causes, and the v7-fix
config that should resolve them.

## 1. v7 eval data (best_val ep54 + final ep99 on val 1304 clips)

| metric | best_val | final | v6 baseline (v11 labels) |
|---|---:|---:|---:|
| target overall L2 | **21.66 cm** | 21.55 cm | ~5–10 cm |
| target <5cm hit | 4.5 % | 6.3 % | ~30–40 % expected |
| target <10cm hit | 17.5 % | 19.2 % | ~60–70 % expected |
| target hand L2 (per-part) | 22–23 cm | 22–23 cm | ~7–10 cm |
| target foot L2 | 25–26 cm | 25–26 cm | ~5–8 cm |
| target pelvis L2 | 15.4 cm | 15.4 cm | ~10 cm |
| contact macro_f1_per_part | 0.195 | 0.207 | ~0.45 |
| contact any_part_f1 | 0.379 | 0.403 | ~0.55 |
| phase macro F1 | **0.628** | 0.617 | ~0.50 |
| support macro F1 | 0.411 | 0.405 | ~0.35 |

Phase / support actually IMPROVED vs v6 (sharper v12 contact boundaries
help phase head). Contact head retreated (sparser positives). **Target
head broke catastrophically** — this is the blocker.

## 2. Root causes

### Cause A — Sparse contact gating wastes target supervision

`PredictorLoss` (legacy v6 design) gates target-xyz regression by
`gt_contact > 0.5`:

```python
contact_gate = (gt_contact > self.contact_threshold).float() * frame_mask.unsqueeze(-1)
loss_target = (loss_target_per_cell * contact_gate).sum() / contact_gate.sum()
```

But `contact_target_xyz_gt` (closest-surface-point per body part per
frame) is emitted by `extract_target.py` **for every (frame, part) cell**
— 100 % non-zero in v12 npz files (verified locally on a chairs clip).
The semantic is "if this body part wanted to touch the object at this
moment, the closest surface point would be at xyz".

| label set | mean contact frame frac | target supervision lost |
|---|---:|---:|
| v11 | ~70 % | ~30 % |
| v12 strict | ~50 % | **~50 %** |

So v7 lost an additional ~20 % supervision relative to v6 just from
the v12 contact-frac drop, on top of v6's 30 % already wasted.

### Cause B — Kendall multi-task weights mis-adapted

Wandb trajectory:

| epoch | loss_target | log_var_target | weight_target = exp(-log_var) |
|---:|---:|---:|---:|
|   1 | 0.132 | -0.05 |  1.05 |
|  21 | 0.029 | -2.81 | 16.6  |
|  51 | 0.025 | -3.00 | 20.1  |
|  99 | 0.021 | -3.15 | **23.4** |

Kendall-CVPR'18 reads "loss_target value is small, task must be
well-balanced" and pushes the weight up. But the small numerical loss
is an artefact of smooth-L1 on a 21 cm error: smooth-L1(0.21 m) ≈
0.5 × 0.21² = 0.022, exactly matching the converged value. Kendall
mistakes "small loss" for "task converged".

With supervision already sparse (Cause A), the boosted weight can't
drive the L2 down further — there aren't enough samples to push the
solution off the bad fixed point. So Kendall's auto-balancing was
counter-productive in the v12-strict regime.

### Cause C — smooth-L1 saturates gradient at large errors

For |error| > 1 m (Huber β=1 default), smooth-L1's gradient is constant
± 1, ignoring the error magnitude. For PIANO body-to-mesh distances in
metres, all "wrong" predictions live in the linear regime where gradient
direction is right but magnitude doesn't scale with error. Combined
with sparse supervision, outliers drift unrecoverable.

### Cause D — Per-part eval display bug (not a training issue)

The eval JSON's `per_body_part.support` field is the contact head's
TP+FN count, not the target gating count (45,383 hand cells = total
hand contact-positive frames, not target-gated cells). The
`per_body_part.mean_l2_m` numbers ARE real and consistent with the
overall 21.66 cm. Eval display bug; doesn't change the diagnosis.

## 3. Other observations (sanity, not root cause)

- **val_loss reads 0.0000 on certain csv rows**: read artefact —
  wandb logged val every 5 epochs at step=epoch+1, but train logs at
  intra-epoch step indices (1, 11, 21...). val rows 5, 10, 15... DO
  have non-zero values. val_every_epochs IS firing correctly.
- **n_contact_frames per logged step ≈ 3300**: low because v12 strict
  contact frac is ~50 % vs v11 ~70 %, and per-step batch averaging
  reflects this.
- **Phase head IMPROVED** (macro F1 0.628 vs v6 ~0.50): v12's sharper
  contact boundaries make phase boundaries cleaner — silver lining.

## 4. v7-fix design

Three orthogonal fixes, ordered by intervention strength:

| fix | v7 setting | v7-fix setting | rationale |
|---|---|---|---|
| **A** Pin task weights manually | `use_kendall_weights: true` | `false` | Don't let Kendall mis-adapt under sparse supervision |
| **B** Re-balance fixed weights | (auto) | `target_weight: 5.0` | Compensate for losing the auto-balancer; matches v6's effective ratio |
| **C** Supervise target everywhere | `target_gate_kind = "contact"` (legacy implicit) | `"all"` | Recover the ~50 % wasted supervision; closest-surface-point is well-defined regardless of contact state |

Fix C is the most architectural change. Justification:
- `extract_target.py::extract_contact_target` returns
  `contact_target_xyz_gt: (T, B, 3)` — closest-mesh-point for **every**
  (frame, body_part), independent of contact state.
- Inference still uses contact_state to gate WHERE the target xyz is
  consumed (z_int conditioning); training "all-frame" supervision is
  separate from inference-time gating.
- Trade-off: for non-contact frames, the model now learns to predict
  closest-surface-point even when the body isn't reaching for the
  object. This is geometric prediction, not contact prediction —
  arguably a better-defined task than the original gated objective.

## 5. Predicted v7-fix outcome

Best estimate:

| metric | v7 | v7-fix predicted |
|---|---:|---:|
| target overall L2 | 21.7 cm | **6–10 cm** |
| target <5cm hit | 4.5 % | 25–40 % |
| target <10cm hit | 17.5 % | 55–70 % |
| contact macro F1 | 0.20 | 0.20–0.25 (~unchanged) |
| phase macro F1 | 0.628 | 0.55–0.65 (~unchanged) |
| support macro F1 | 0.411 | 0.40–0.45 (~unchanged) |

**Acceptance**: target overall L2 < 12 cm → unblocks Stage B v18 launch.

If v7-fix still produces L2 > 15 cm:
1. Suspect smooth-L1 saturation (Cause C); switch target loss to L2.
2. Suspect supervision frame-count itself; consider mixed-batch upsampling
   of contact-positive cells.
3. Suspect predictor capacity; this is unlikely (architecture unchanged
   and v6 worked).

## 6. Implementation summary

- `src/piano/training/losses.py`: `PredictorLoss` accepts
  `target_gate_kind: "contact" | "all"`. Default `"contact"` preserves
  v6 behaviour.
- `src/piano/training/train_predictor.py`: reads
  `cfg.loss.target_gate_kind` (default `"contact"`).
- `configs/training/predictor_v7fix_v12strict.yaml`: applies all 3 fixes
  on top of v7's data path. `output_dir =
  runs/training/predictor_v7fix_v12strict`, doesn't conflict with v7.

Local sanity test passed:
- `target_gate_kind="contact"` matches legacy v6 numerically
- `target_gate_kind="all"` gives non-zero loss even with all-zero
  `gt_contact`
- Invalid `target_gate_kind` raises ValueError.

## 7. Next-step gating

| outcome of v7-fix eval | next step |
|---|---|
| target L2 < 12 cm AND contact macro_f1 ≥ 0.20 | **Launch Stage B v18 immediately** |
| target L2 12–15 cm | Add Cause C fix (smooth-L1 → L2); ~6 h server retrain |
| target L2 > 15 cm | Deeper investigation; possibly batch upsampling or larger predictor capacity |

If Stage B v18 launches with a v7-fix predictor that has L2 < 12 cm,
predicted v18 outcome (per analyses/2026-05-03_pseudo_label_v12_strict_design.md):
- raw correct_part_recall: 0.176 → 0.30+
- guided correct_part_recall: 0.292 → 0.40+
- Visual: real contact, not approach.

## 8. References

- v7 eval data: `runs/eval/predictor_v7_v12strict_val_best.json`,
  `runs/eval/predictor_v7_v12strict_val_final.json`
- v7 trajectory: `runs/wandb_logs/wandb_history_predictor_v7_v12strict.csv`
- Legacy contact gate: `src/piano/training/losses.py::PredictorLoss`
- target xyz extraction: `src/piano/data/pseudo_labels/extract_target.py`
- v12 strict design: `analyses/2026-05-03_pseudo_label_v12_strict_design.md`
- v7 train config: `configs/training/predictor_v7_v12strict.yaml`
- v7-fix train config: `configs/training/predictor_v7fix_v12strict.yaml`
- Implementation commit: `32dc2b5`
