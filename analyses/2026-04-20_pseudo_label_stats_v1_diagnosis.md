# Pseudo-Label v1 Stats — Diagnosis and Recalibration — 2026-04-20

## Context

First full rerun after P0 fixes (fps propagation + deterministic patch
atlas, commit `9d11f1a`). Stats computed post-hoc with
`piano-pseudo-label-stats` (commit `34ccf3c`). Four subsets, 8473
sequences evaluated. Source label `.npz` stamped 2026-04-20 00:25-05:25
UTC.

## Results (v1 stats)

| subset | zero-contact seqs | any-part frame rate | sitting frame rate | manipulation reached | mean target entropy |
|---|---:|---:|---:|---:|---:|
| chairs | 81.2% | 2.8% | 0.0% | 10.7% | 0.003 / 2.773 |
| imhd | 92.7% | 0.5% | 0.0% | 5.9% | 0.008 / 2.773 |
| neuraldome | 90.6% | 1.1% | 0.0% | 6.5% | 0.002 / 2.773 |
| omomo_correct_v2 | 98.8% | 0.1% | 0.0% | 1.1% | 0.001 / 2.773 |

Every subset fires most of its quality flags. The label set is
**unusable for training** in this state.

## Diagnosis

The P0 fixes were correct but did not touch the root cause. The
extraction is broken by three distinct bugs that the stats make legible:

### Bug 1: uniform 2 cm distance threshold is wrong for joint-based contact

`extract_contact` queries the distance from each tracked *joint* to
the object mesh and thresholds at 2 cm. But SMPL joint centers sit
inside the body, not on the skin:

| Body part | Joint location | Offset to contact surface |
|---|---|---|
| left/right_hand (wrist, idx 20/21) | inside forearm | 5-8 cm to palm |
| left/right_foot (ankle, idx 7/8) | ankle bone | 8-10 cm to sole |
| pelvis (idx 0, SMPL root) | inside abdomen | 15-20 cm to seat during sitting |

The object-convention check run on a chairs sequence confirmed this
directly: across the 3 inspected frames of a clearly-seated clip,
pelvis-to-mesh distance was 18-20 cm — the joint never came within
2 cm of the mesh, so pelvis contact never fired. Every chairs
sequence (1502/1502) shows `sitting` support at 0.0% for the same
reason (support inherits from pelvis contact).

### Bug 2: velocity gating multiplies the contact score

```python
contact[:, bp_idx] = dist_score * vel_score
```
with `velocity_threshold = 0.1 m/s`. Even when `dist_score` is high
(hand at 1 mm from mesh at some frame of the convention-check seq),
any body-part motion drops `vel_score` to near zero and the product
goes to zero. World-frame speed on a moving object is the wrong
quantity in the first place — that's the manipulation-vs-stable
distinction, which already belongs to `phase`.

### Bug 3: target soft-assign kernel collapses to hard

`soft_patch_assignment` built logits as `-d / (2σ²)` (linear distance
in the exponent) with `sigma = 0.01`. For typical InterAct patch
spacing (15-30 cm) that makes the softmax essentially argmax. Measured
entropy across all 8473 contact frames: 0.003 / ln(16) = 2.773 max.
100% of sequences with contact have degenerate targets. The review
(`SUGGESTION.md` P2) flagged this and was right about both the
severity and the fix.

Bugs 1 + 2 compound: bug 1 makes most frames distance-ineligible;
bug 2 zeroes the few that remain. That produces the 81-99%
zero-contact observations. Bug 3 sits downstream and would have
broken target supervision even if 1 + 2 were fixed.

## Fixes (commit `d641732`)

1. Per-body-part distance thresholds (meters):
   ```
   left_hand:  0.08      right_hand: 0.08
   left_foot:  0.12      right_foot: 0.12
   pelvis:     0.20
   ```
   `distance_sigma` raised from 0.005 to 0.03 for a realistic
   transition width.

2. `use_velocity_gating` default False. Kept as an optional ablation
   flag with loosened defaults (threshold 0.5 m/s, sigma 0.2 m/s).

3. `soft_patch_assignment` kernel changed to `-(d²) / (2σ²)`; default
   `sigma` raised from 0.01 to 0.05. `TargetConfig.soft_sigma` default
   updated to match.

No API breakage: `ContactConfig` still accepts the legacy fields,
they're just unused unless `use_velocity_gating=True`.

## Expected outcome after rerun

These are the numbers I expect to see, with ±loose tolerance, after
re-extracting all four subsets. They define the pass/fail bar for the
next rerun:

| subset | zero-contact seqs | sitting frame rate | manipulation reached | target entropy (mean) |
|---|---:|---:|---:|---:|
| chairs | < 30% | > 25% | > 30% | > 1.2 |
| imhd | < 30% | ~0% (no sitting in imhd) | > 30% | > 1.2 |
| neuraldome | < 40% | ~0% (mostly manipulation) | > 40% | > 1.2 |
| omomo_correct_v2 | < 30% | ~0% | > 50% | > 1.2 |

`chairs sitting > 25%` is the single most informative number: if
it's above 10% we've broken the real pelvis-contact bottleneck; if
it's above 25% we've also set the threshold correctly. If sitting
is still < 5%, the pelvis threshold needs to go up further (try 0.25).

## Action Items (→ PLAN.md)

- [x] Land the recalibration (`d641732`).
- [ ] Re-run pseudo-label extraction on server:
      `bash scripts/data/rerun_pseudo_labels_interact.sh`.
- [ ] Re-run stats: `bash scripts/server/pseudo_label_stats.sh`.
- [ ] Review stats against the expected-outcome bar above; promote to
      training only if every row passes.
- [ ] If `sitting` still < 10% for chairs, raise `pelvis` threshold to
      0.25 and re-check. Beyond that we're in "need body-vertex
      contact instead of joint contact" territory (P1 in Codex review;
      not a threshold problem).
- [ ] Decide on remaining P1/P2 items (extra joints, closest-surface
      target, median→majority filter) after the rerun numbers land.

## Lessons

- "Distance < ε" as a contact proxy has to be calibrated to whatever
  the distance is actually measuring. For joint-center data, the
  anatomical offset matters and is non-trivial.
- Convention checks (geometry correctness) pass independently of
  threshold correctness. Would be worth a separate sanity test that
  reports each body-part's per-frame distance *distribution* on a
  labelled reference clip, so threshold mis-calibration shows up
  immediately instead of emerging from aggregate stats after a 4-hour
  rerun.
