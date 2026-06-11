# Round-42 condition 2x2 diagnostic — verdict and R43 plan

- Date: 2026-06-11
- Author: Claude Code (verdict), Codex (R42 2x2 launcher + summary)
- Source artifacts:
  - `analyses/round42_cond_2x2_20260611_205914_manifest.txt`
  - `analyses/round42_cond_2x2_20260611_205914/round42_cond_2x2_summary.md`
  - `analyses/round42_cond_2x2_20260611_205914/diag/{oo,go,og,gg}/*/sustained_contact_summary.md`
- Context: user hypothesized Stage-1.5 OOD on generated Stage-1 may be
  the real R41 bottleneck. Codex built the 2x2 diag to test it.

---

## TL;DR

User's hypothesis is **confirmed and stronger than expected.**

| cell | sustained drift mean | LH | RH | pelvis | track frac mean |
|---|---:|---:|---:|---:|---:|
| OO oracle S1 + oracle C41/S4 | **7.52** | 11.5 | 14.0 | 3.2 | 1.01 |
| OG oracle S1 + generated C41/S4 | 11.94 | 17.8 | 26.0 | 5.5 | 1.07 |
| GO generated S1 + oracle C41/S4 | 16.98 | 30.8 | 31.7 | 4.3 | 0.95 |
| **GG generated S1 + generated S1.5** | **39.34** | 52.5 | 51.6 | **33.9** | **0.55** |

- **PB1 oracle floor (OO)** = 7.52 cm. This is the irreducible drift
  of the PB1 model itself; no upstream change can fix it.
- **GO (R41's actual setup)** = 16.98 cm. R41 5-cell ablation
  (A0-A4: 17.04-18.00 cm) matches GO almost exactly — R41 was just
  re-measuring this cell with slightly different cascade losses.
- **GG (true deployment)** = 39.34 cm. This is **2.3× worse than GO
  and 5.2× worse than OO.** The actual user-facing pipeline is far
  worse than any single-stage measurement suggested.

---

## Where the R41 verdict was wrong

R41 trained 5 Stage-1 cells against frozen PB1 with **GT-derived
C41/S4** in the cascade loop. That is exactly the GO cell of this 2x2.

R41 A0-A4 drift_max:

| cell | drift_max mean |
|---|---:|
| A0 | 18.00 |
| A1 | 17.06 |
| A2 | 17.04 |
| A3 | 17.20 |
| A4 | 17.64 |
| **R42 GO** | **16.98** |

Identical ± 1 cm. The R41 ablation was an expensive way to confirm
that the GO ceiling exists.

**R41 cannot improve full-cascade GG because R41 never sees generated
C41/S4 during training.** Stage-1.5 is held frozen at an oracle-trained
checkpoint and never adapts to Stage-1's actual output distribution.

---

## What the per-part breakdown shows

### GG track-fraction is broken

| part | OO | GO | OG | GG |
|---|---:|---:|---:|---:|
| LH track frac mean | 1.04 | 0.99 | 0.99 | **0.64** |
| RH | 1.05 | 0.94 | 1.08 | **0.51** |
| LF | 0.92 | 0.58 | 1.52 | 0.89 |
| RF | 1.07 | 1.18 | 1.16 | **-0.26** |
| pelvis | 0.99 | 1.03 | 0.91 | **0.57** |

GG right_foot track frac = **-0.26** is physically impossible for an
"approximately following" signal — it means the foot moves
**anti-correlated** with the object on average. Stage-1.5 fed
generated Stage-1 is producing inverted phase/contact signals that
PB1 then commits to.

### GG low-track fraction (% segments with track < 0.5)

| part | OO | GO | OG | **GG** |
|---|---:|---:|---:|---:|
| LH | 2.2% | 4.3% | 8.7% | **47.8%** |
| RH | 0.0% | 5.4% | 8.1% | **45.9%** |
| LF | 16.0% | 32.0% | 4.0% | **64.0%** |
| pelvis | 5.1% | 3.4% | 6.8% | **49.2%** |

Roughly **half of all GG segments completely lose the object**. This
is the true deployment failure mode — not "drifts a bit further", but
"hands and pelvis stop tracking entirely on ~50% of clips".

### GG pelvis drift jumps 11× over OO/GO/OG

OO/GO/OG pelvis drift is 3.2-5.5 cm (normal). GG pelvis drift is
**33.91 cm**. Stage-1.5 in OOD mode produces world-frame root motion
that PB1 cannot stabilize.

---

## Why Stage-1.5 fails OOD

Stage-1.5 was trained on **GT** Stage-1 coarse (23-D oracle). At
deployment it gets **R41-generated** Stage-1 coarse. Three failure
modes likely compound:

1. **Distribution mean gap.** Generated Stage-1 has slightly different
   per-channel mean/std than GT (we saw similar gaps in R31 V2 closure
   for V8 V6 vs PB1-train-time GT+σ=0.05 noise). Stage-1.5 has zero
   training-time σ-augmentation; even small mean shifts route through
   its first-layer projection.
2. **Phase/footstep prediction collapses.** S4 includes phase and
   footstep channels that depend on global gait timing. If generated
   Stage-1 has slightly different pelvis dynamics, Stage-1.5 mispredicts
   gait phase, which is what produces the inverted right-foot tracking
   in GG.
3. **C41 root trajectory amplification.** Small Stage-1 root errors
   become larger Stage-1.5 root errors which become catastrophic PB1
   root errors. Stage-1.5 is supposed to refine Stage-1's root, not
   amplify its noise.

---

## R43 plan — priority ordered

### P0: Retrain Stage-1.5 on R41-generated Stage-1 distribution

**Hypothesis:** OG = 11.94 cm shows Stage-1.5 is fundamentally fine
when fed clean S1 — it adds ~4 cm of refinement noise but doesn't
collapse. Retraining Stage-1.5 with R41-generated Stage-1 as input
should pull GG closest to GO (16.98 cm) or better.

**Concrete steps:**

1. Pick the R41 winner by GO-metric (A2 world_vel had marginally
   best GO 17.04 cm; pick A2 as the Stage-1 source).
2. Generate full Stage-1 coarse cache for train+val from A2's final.pt.
3. Retrain Stage-1.5 (R38-B1 architecture) on this cache, keeping
   GT C41/S4 as the target.
4. Run 2x2 again. New GG should land in [12, 17] cm range.

Cost: 1× Stage-1.5 training (~3-4 h at R38-B1 config), 1× R42 2x2
re-run (~30 min). Total ~5 h.

### P1: Cascade-train Stage-1 + Stage-1.5 jointly with frozen PB1

Same R41 trainer skeleton but Stage-1.5 also has gradient flow. Both
upstream predictors adapt to each other and to PB1's actual input
distribution simultaneously.

Cost: 1× joint cascade training (~6-8 h, ~2× R41 cost due to
backprop through both Stage-1 and Stage-1.5). Add to roadmap after P0
result tells us whether Stage-1.5-only refit is sufficient.

### P2: σ-augmentation on Stage-1.5's S1 input during training

Add `stage1_coarse_noise_std=0.05` (matching PB1 training) to
Stage-1.5 trainer. Cheaper than P0 (no need to generate a
distribution-matched cache) but addresses only mean-gap, not
distribution-shape mismatch.

Use as fallback if P0 retraining doesn't converge well.

### Explicitly deprecated

- **More R41-style Stage-1-only cascade ablations.** GO ceiling at
  ~17 cm is established. Adding losses to Stage-1 cascade cannot
  break that ceiling without changing Stage-1.5.
- **Anchor-loss tuning.** R41 A4 (anchor) was the worst cell. Anchor
  signal flowing through frozen PB1 through Stage-1's 23-D bottleneck
  doesn't reach hand placement.

---

## What this re-frames for the project

Before R42 2x2: "Stage-1 is the bottleneck; cascade-train it to
match downstream PB1." Five R41 cells, ~3 days of GPU time, no
improvement.

After R42 2x2: "Stage-1.5 OOD on generated Stage-1 amplifies upstream
noise by ~3-5× and produces ~50% complete tracking failures." Single
2x2 diag (~30 min), zero training time, definitive separation of
blame.

The lesson: when a multi-stage pipeline underperforms, **measure each
condition pair before training**. R41's 3 days of GPU should have
been one R42 2x2 first.

Memory note: this should land in feedback memory as "before training
fix for stage X, run condition-ablation diag to confirm stage X is
actually the bottleneck."
