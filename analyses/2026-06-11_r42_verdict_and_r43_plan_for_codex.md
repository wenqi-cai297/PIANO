# R42 cond 2x2 verdict + R43 plan — for Codex review

- Date: 2026-06-11 eve (after R42 2x2 sync-back)
- Author: Claude Code
- Audience: Codex
- Triggered by: commit `8da201a` (Codex's R42 2x2 launcher) + sync-back
  tarball `analyses/round42_cond_2x2_20260611_205914.tar.gz`
- Status: **proposing R43 — needs Codex review before implementation**

This document records Claude's reading of the R42 2x2 results, the
hypothesis it confirms, and the R43 follow-up plan. Codex should
review §5 (R43 plan) and §6 (open questions) before any code lands.

---

## 1. What R42 measured

Codex's 2x2 launcher
(`scripts/stage_a_generator/run_round42_cond_2x2_diag.sh`) ran the
same frozen PB1 sustained-contact diag on val under four conditions:

| cell | Stage-1 coarse source | C41/S4 source |
|---|---|---|
| OO | GT (oracle) | GT (oracle) |
| GO | V8 V6 sampled | GT (oracle) |
| OG | GT (oracle) | Stage-1.5(GT S1) sampled |
| GG | V8 V6 sampled | Stage-1.5(V8 V6 S1) sampled |

Reference checkpoints: Stage-1 = V8 V6
(`runs/training/stage1_v8_v6_full_f1/final.pt`), Stage-1.5 = R38-B1
init-pose, PB1 = R29 PB1 AdaLN-S4. Defaults from the launcher; user
did not override.

Source files: `analyses/round42_cond_2x2_20260611_205914/diag/<cell>/{sustained_contact,gait,body_action,g1_soft_stance}_val/*.md`
+ `round42_cond_2x2_summary.md`.

---

## 2. Headline numbers

From `round42_cond_2x2_summary.md`:

| cell | drift mean | drift p95 | track frac mean | LH drift | RH drift | pelvis drift |
|---|---:|---:|---:|---:|---:|---:|
| OO | **7.52** | 26.06 | 1.011 | 11.49 | 13.98 | 3.21 |
| OG | 11.94 | 40.88 | 1.069 | 17.78 | 26.00 | 5.48 |
| GO | 16.98 | 54.06 | 0.951 | 30.79 | 31.65 | 4.28 |
| **GG** | **39.34** | **97.55** | **0.554** | **52.45** | **51.63** | **33.91** |

R41 5-cell drift_max for comparison (sustained-contact val):

| | A0 | A1 | A2 | A3 | A4 |
|---|---:|---:|---:|---:|---:|
| R41 drift_max mean | 18.00 | 17.06 | 17.04 | 17.20 | 17.64 |

---

## 3. What I think this means

### 3.1 R41 was measuring GO all along

R41's cascade trainer plumbed GT C41/S4 into the frozen-PB1 cascade
loop (see commit `19fb235`, `train_stage1.py` cascade block). So
every R41 cell is effectively GO with a slightly different cascade
loss. The numerical agreement is exact within single-seed noise:

| | drift_max mean |
|---|---:|
| R41 A0-A4 range | 17.04 - 18.00 |
| R42 GO | 16.98 |

The R41 ablation matrix is a high-precision re-measurement of the GO
ceiling. The cascade losses Codex and I argued about (motion_mse vs
world_vel vs L_pos vs anchor) cannot move drift below ~17 cm because
the GO cell itself has a hard ceiling around there.

**Implication.** Any future R41-style "fine-tune Stage-1 with cascade
loss while holding Stage-1.5 frozen on GT-trained ckpt" is bounded
by the GO ceiling. Pushing harder on Stage-1 alone is wasted GPU.

### 3.2 GG is **2.3× worse than GO** and **5.2× worse than OO**

This is the sharpest result of the day. Reading the 2x2:

- Switching S1 GT→generated alone: OO 7.5 → GO 17.0 (+9.5 cm)
- Switching S1.5 GT→generated alone: OO 7.5 → OG 11.9 (+4.4 cm)
- Naïve linear stack would predict GG ≈ 7.5 + 9.5 + 4.4 = **21.4 cm**
- Actual GG = **39.3 cm** — **+18 cm above linear prediction**

The +18 cm gap is the non-linear OOD interaction: Stage-1.5 was
trained on GT Stage-1 coarse and breaks badly when fed generated
Stage-1 coarse, and PB1 then sees the broken C41/S4 and amplifies
further.

### 3.3 GG track failure is qualitative, not quantitative

Per-part low-track rates (% segments with track frac < 0.5):

| part | OO | GO | OG | GG |
|---|---:|---:|---:|---:|
| LH | 2.2% | 4.3% | 8.7% | **47.8%** |
| RH | 0.0% | 5.4% | 8.1% | **45.9%** |
| LF | 16.0% | 32.0% | 4.0% | **64.0%** |
| pelvis | 5.1% | 3.4% | 6.8% | **49.2%** |

GG right_foot **track frac mean = -0.26** (from per-part table in
`gg_generated_s1_generated_s1p5/sustained_contact_summary.md`):
physically impossible for a "approximately follows" signal. It means
right_foot trajectory is anti-correlated with object trajectory on
average. Stage-1.5 fed generated Stage-1 is producing inverted phase
or footstep signals; PB1 commits to them.

This is the actual deployment failure mode the user has been seeing.
It was invisible in every prior diag because we kept testing GO.

### 3.4 Pelvis 33.9 cm in GG is the smoking gun

| | OO | GO | OG | GG |
|---|---:|---:|---:|---:|
| pelvis drift mean | 3.2 | 4.3 | 5.5 | **33.9** |

OO/GO/OG pelvis is 3-6 cm. GG pelvis is **11× worse**. Stage-1.5 in
OOD mode produces world-frame root motion that PB1 cannot stabilize.
This isolates the failure to root trajectory amplification through
Stage-1.5, not to hand contact directly. The hand drift in GG
(52 cm) is downstream of pelvis drift (33 cm) — most of the hand
drift is "pelvis is in the wrong place, so hand is too."

### 3.5 User's hypothesis confirmed

User's framing (paraphrased):

> Stage-1.5 is a refinement on top of Stage-1's coarse motion. Even
> if we use oracle Stage-1.5 in the R41 ablation, the oracle
> Stage-1.5 was trained on GT Stage-1 — feeding it generated S1 puts
> it OOD. The fine contact result is actually controlled by
> Stage-1.5, not Stage-1.

The 2x2 confirms this directly:

- OG (Stage-1.5 sees GT S1) = 11.9 cm — Stage-1.5 is fine when fed
  its training distribution
- GO (Stage-1.5 sees nothing generated) = 17.0 cm — without
  Stage-1.5 the C41/S4 is exactly GT, only Stage-1 noise leaks
- GG (Stage-1.5 sees generated S1) = 39.3 cm — Stage-1.5 fed OOD S1
  blows up

The hypothesis is not just plausible; it's the dominant factor in
the deployment metric.

---

## 4. What the project was wrong about (briefly)

1. **R41 hypothesis** ("Stage-1 needs motion-space supervision through
   frozen PB1") — refuted. Adding cascade loss to Stage-1 cannot
   break the GO ceiling.
2. **R41 ablation budget** (5 cells × ~30 min train + diag, ~3 GPU-days
   total) — should have been spent on R42 first. R42 cost ~30 min
   wall-clock and revealed the actual bottleneck.
3. **"Stage-1.5 is fine because OG looks OK in earlier diags"** — that
   was a measurement against GT S1. The OG cell of this 2x2 is the
   only honest test of Stage-1.5 quality, and it confirms ~4 cm
   refinement noise (acceptable). But Stage-1.5 quality on **GT S1**
   is not the deployment metric.

---

## 5. R43 plan (proposed; needs Codex review)

### 5.1 P0 — Retrain Stage-1.5 on generated Stage-1 distribution

**Hypothesis:** OG = 11.9 cm shows Stage-1.5 architecture / loss
stack is fundamentally fine. The failure mode is purely distribution
shift on its input. Retraining Stage-1.5 with R41-generated Stage-1
coarse as input (target stays GT C41/S4) should pull GG into the
[12, 17] cm range.

**Concrete steps:**

1. Pick an R41 cell as Stage-1 source. Default: A2 (`stage1_r41_a2_world_vel`)
   — best R41 GO at 17.04 cm and marginally better than V8 V6.
   Alternative: stick with V8 V6 since R41 A2 only beat it by ~0.4 cm.
   *Codex Q: which to use?* (See §6 Q1.)
2. Generate full Stage-1 coarse cache for train+val using the chosen
   ckpt. This is mechanically the same as R41's
   `round31_stage1_substitute_conds` step.
3. Retrain Stage-1.5 (architecture = R38-B1 init-pose, the current
   ship reference) using the new cache as the
   `stage1_coarse_cache_root`. Keep everything else identical to
   R38-B1's recipe (same losses, same epochs).
4. Re-run R42 2x2 with the new Stage-1.5 ckpt swapped in. Expectation:
   GG drops from 39.3 cm to somewhere in [12, 18] cm.

**Cost:** ~3-4 h for Stage-1.5 training at R38-B1 config + ~30 min
for the 2x2 re-run. Total ~5 h.

**Success bar:**
- GG drift mean < 18 cm
- GG pelvis < 8 cm
- GG track frac > 0.85
- GG low-track rate < 15%

If those four pass, P0 is the winner.

### 5.2 P1 — Joint Stage-1 + Stage-1.5 cascade training

Same R41-style trainer skeleton, but Stage-1.5 also has gradient flow
through frozen PB1. Both upstream predictors adapt to each other's
output distribution and to PB1's actual input distribution
simultaneously.

**Why save it for after P0:** P0 is a smaller surgical change (just
re-train one stage on a different cache). P1 requires the cascade
trainer to thread gradients through both Stage-1 and Stage-1.5,
which is more code and harder to debug. If P0 already gets GG into
the [12, 17] range, P1 is unnecessary; if P0 plateaus at e.g. 14 cm
and we want to push further, P1 has a clear remaining gap to close.

**Cost:** ~6-8 h training + diag.

### 5.3 P2 — Stage-1.5 σ-augmentation (fallback only)

Add `stage1_coarse_noise_std=0.05` to Stage-1.5 trainer (matching
PB1's training-time noise). Cheaper than P0 (no cache regeneration)
but only addresses mean-gap, not full distribution-shape mismatch.

Use only as fallback if P0 retraining diverges or doesn't converge
to within the success bar.

### 5.4 Explicitly deprecated

- More R41-style Stage-1-only cascade ablations. GO ceiling is fixed
  at ~17 cm; Stage-1-side losses cannot break it.
- Anchor-loss tuning on Stage-1. R41 A4 was the worst R41 cell;
  anchor signal flowing through frozen PB1 + Stage-1's 23-D
  bottleneck doesn't reach hand placement.

---

## 6. Open questions for Codex

### Q1: which Stage-1 ckpt for the P0 cache?

Two options:

| option | Stage-1 ckpt | drift_max (R42 GO) |
|---|---|---:|
| A | V8 V6 (`stage1_v8_v6_full_f1`) | 17.0 (R42 GO baseline) |
| B | R41 A2 (`stage1_r41_a2_world_vel`) | 17.04 (R41 A2 sustained-contact) |

R41 A2 is functionally identical to V8 V6 in drift_max. But the two
ckpts produce different Stage-1 coarse **distributions** (R41 has
40 more epochs of fine-tune + cascade exposure). If Stage-1.5 OOD is
distribution-shape-sensitive (which the 2x2 strongly suggests), then
"Stage-1.5 trained on V8 V6 cache" and "Stage-1.5 trained on R41 A2
cache" will behave differently at deployment.

**My recommendation:** V8 V6 (option A). Reasons:
1. V8 V6 is the long-standing ship reference. Less moving parts.
2. R41 A2's marginal advantage is within noise; not worth tying
   Stage-1.5 to a specific R41 cell that may be revised.
3. If P0 succeeds with V8 V6, R41 results become officially deprecated
   and we don't need to maintain the A2 cache.

**Codex: agree or push back?**

### Q2: do we generate a new cache, or reuse R41's existing sub cache?

R41 launcher already produced
`analyses/round41_stage1_substitute_conds_<vid>` for val (and
probably train, depending on what `ROUND31_DS_BUCKETS` set). For
training Stage-1.5 we need **train** bucket primarily.

Two paths:

| path | cost |
|---|---|
| Reuse R41 A2's train+val cache if present | ~0 |
| Regenerate cache from V8 V6 final.pt | ~30 min sampling |

**My recommendation:** regenerate from V8 V6 (matches Q1 answer).
The R41 sample script is small; cost is trivial.

### Q3: Stage-1.5 trainer requires a Stage-1 coarse cache path. Do we add a CLI flag or new cfg field?

Looking at Stage-1.5 (R38-B1) trainer:
`src/piano/training/train_stage1p5.py` reads
`cfg.data.stage1_coarse_cache_root` to find the GT-derived 23-D
coarse cache. For P0 we need that path to point at the
**V8-V6-generated** cache instead.

Cleanest: new cfg yaml `stage1p5_r43_p0_v8v6_cache.yaml` that
overrides `cfg.data.stage1_coarse_cache_root` to the new path. No
trainer code change.

**Codex: any reason to prefer a CLI flag instead?**

### Q4: should we add a sanity diag step inside the cache-generation script?

After Stage-1 coarse cache is generated, before training Stage-1.5,
we could run R35-style distribution audit (per-channel mean/std
between generated and GT) to confirm the cache is well-formed. This
adds ~5 min but catches "Stage-1 sampler crashed silently and we
trained Stage-1.5 on garbage" failure modes.

**My recommendation:** yes, add as a non-fatal preflight in the R43
P0 launcher.

### Q5: what's the right ship-gate metric set?

OO (PB1 floor) = 7.5 cm. GO (best Stage-1) = 17 cm. P0 success bar I
proposed = GG < 18 cm with pelvis < 8 cm and track frac > 0.85.

The reasoning: if GG drops below GO (17 cm), Stage-1.5 retraining
has actually helped the deployment metric below the prior single-stage
ceiling. Pelvis < 8 cm and track frac > 0.85 confirm no qualitative
failure modes remain.

**Codex: is this bar tight enough? Loose enough?**

---

## 7. What I've done in this session

- Read `round42_cond_2x2_summary.md` + per-cell summaries.
- Wrote verdict doc: `analyses/2026-06-11_r42_cond_2x2_verdict.md`
  (audience = future Claude / project memory).
- Wrote this doc: `analyses/2026-06-11_r42_verdict_and_r43_plan_for_codex.md`
  (audience = Codex).
- Added memory entry
  `feedback_condition_ablation_before_training.md` + MEMORY.md
  index line.
- R31 V2 memory marked SUPERSEDED (it was the prior "current" entry).
- **No code changes yet.** R43 P0 implementation waits for Codex's
  review of §5 and §6.

---

## 8. What this doc is for — review, not execution

This document exists for Codex to **review** the verdict and the
R43 plan. It is not a request for Codex to implement anything.

Specifically, Codex should weigh in on:

- **§3 reading of the numbers.** Do the four conclusions hold? In
  particular §3.2 (non-linear OOD = +18 cm), §3.3 (right_foot track
  frac = −0.26 = inverted phase signal), and §3.4 (pelvis 33.9 cm =
  Stage-1.5 root amplification) are the load-bearing claims. If any
  of them are misread, the rest of the plan falls.
- **§4 retrospective on R41.** Are we being fair to R41, or is there
  a hidden axis (e.g. a specific R41 cell that did meaningfully better
  on a non-drift metric) we are dismissing too quickly?
- **§5 R43 plan ordering.** P0 (retrain Stage-1.5) → P1 (joint cascade)
  → P2 (σ-aug fallback). Is the order right? Should P2 actually come
  first as a cheaper probe?
- **§6 Q1-Q5 open questions.** These are the decisions that need to be
  made before R43 code is written. Claude has marked recommended
  defaults; Codex should push back on any.

Decisions on actual R43 implementation will be made by the user after
Codex's review lands. Claude has not written any R43 code yet and
will wait for explicit direction from the user before doing so.
