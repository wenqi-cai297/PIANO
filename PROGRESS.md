# PIANO Progress

Compact project memory as of 2026-05-01.

## Current Snapshot

Recent Stage B implementation:

```text
v16 alignment-aware contact loss + deterministic mirror-doubled training set
```

Current Stage B best evaluated server checkpoint identifier. The local workspace may only
have eval summaries, not the `.pt` file:

```text
runs/training/generator_v14_sampled_st_contact/best_contact.pt
```

Matched 80-clip contact eval:

| Row | mean_min_dist_per_frame |
|---|---:|
| GT original | 13.09 cm |
| GT VQ roundtrip | 18.47 cm |
| v12 w02 best_val full | 31.82 cm |
| v13 target-trajectory best_val full | 31.57 cm |
| v14 sampled-ST best_contact full | 27.37 cm |
| v14 sampled-ST best_val full | 30.77 cm |
| v14 sampled-ST final full | 31.12 cm |
| v15 alignment-guided best_contact full | 27.62 cm |
| v15 alignment-guided best_contact full_guided | 31.57 cm |
| v12 w02 K=16 distance oracle | 17.93 cm |
| v12 w02 K=16 composite oracle | 18.08 cm |
| v14 best_contact K=16 distance oracle | 16.80 cm |
| v14 best_contact K=16 composite oracle | 17.17 cm |
| v14 best_contact K=64 alignment oracle | 17.92 cm |

Bottom line: v14 is the first clear single-sample contact improvement after
the v12/v13 plateau, and its K=16 candidate pool is also better. v14
best_contact improves full contact from the old `31-32 cm` band to `27.37 cm`;
v14 K=16 distance reaches `16.80 cm`, and v14 K=16 composite re-measures at
`17.94 cm` with moving-coupled `0.3715`, beating the v12 K=16 composite
coupling (`0.351`). The later K64 alignment oracle shows this is still not an
aligned-HOI distribution: spatially close candidates exist, but GT-aligned
part/patch/timing candidates are too rare.

2026-04-30 follow-up analysis: v14 sampled-ST helps both one-shot spatial
contact and K=16 candidate quality, but ordinary single-sample generation still
rarely selects coupled candidates and K64 alignment selection shows the pool
itself lacks enough aligned ones. v15 tested the direct alignment/guidance
branch and did not solve this, so v16 now tests deterministic mirrored-data
doubling before abandoning this loss family.

2026-04-30 visual/alignment update: v14 K=16 composite looks much better than
earlier generations and slightly better than distance-only, but still visibly
fails GT-quality object contact. The new CPU-only diagnostic
`scripts/stage_b_generator/measure_contact_alignment.py` compares generated
samples to GT roundtrip contact in object-local coordinates. Composite K=16 has
moving contact IoU `0.4472`, moving GT-contact recall `0.5438`, correct
GT-body-part recall `0.2378`, and same-GT-part object-local position error
`46.32 cm`. Distance-only is almost identical (`0.4505` IoU, `0.2305` correct
part recall, `46.42 cm` local error). The GT self-check gives IoU, recall, and
correct-part recall `1.0`, plus same-part local position error `0.0`,
validating the diagnostic. Conclusion: the current distance/composite metrics are
insufficient; next guidance must be body-part and contact-target aware.

2026-04-30 K64 alignment-oracle update: selecting among 64 v14 samples with the
alignment-aware score gives oracle contact `17.92 cm` and post-hoc remeasure
`18.71 cm`, but moving-coupled frame fraction drops to `0.3339`. GT-alignment
barely changes in time (`0.4516` moving IoU) and only slightly improves body
part correctness (`0.2496` correct GT-part recall), while same-part local error
improves from `46.32 cm` to `40.30 cm`. Per-candidate capacity is the key
negative result: the best primary alignment error over all K=64 candidates is
still `37.0 cm` on average, and the best moving same-part recall available is
only `0.165`. NeuralDome and OMOMO have zero clips with any K=64 candidate
reaching moving same-part recall >= `0.5`. Conclusion: v14 reranking is close
to exhausted; the distribution itself needs stronger alignment/coupling
training or guidance.

2026-04-30 v15 result update: v15 alignment-guided training is a negative or
at best neutral result. `best_contact` raw full is `27.62 cm`, essentially tied
with v14 `27.37 cm`; the strict GT-alignment readouts are worse than the v14
K-oracle baselines (`0.3804` moving IoU, `0.1684` moving correct GT-part recall,
`55.09 cm` moving same-part local error). Full-RVQ target guidance improves
some temporal-overlap readouts slightly but usually worsens contact/local error:
for best_contact, `full_guided` is `31.57 cm` contact and `59.95 cm` local
error. Local visualization in the `piano` conda env confirms the numbers:
`runs/visualizations/stageB_v0_15_bc_review/{full,full_guided}` still shows
visible human-object offset, and guidance can move the body farther from the
object on trolley/suitcase cases.

2026-04-30 implementation update: v16 has been added for the next server run.
It keeps the v15 alignment objective but changes the data side to deterministic
MoMask/HumanML3D-style mirror doubling: `HOIDataset.__len__` doubles when
`augmentation.mirror_duplicate=true`, even indices are original clips, and odd
indices are forced mirrored copies. Validation/eval remain unaugmented. New
artifacts: `configs/training/generator_v16_alignment_mirror.yaml`,
`scripts/stage_b_generator/run_v16_alignment_mirror.sh`, and
`tests/test_dataset_mirror_duplicate.py`.

2026-05-01 v16 server result: partial positive, doesn't close the K-oracle gap.
Raw `best_contact full` 26.79 cm vs v15 27.62 cm (+0.83 cm); `full_guided`
28.91 cm vs v15 31.57 cm (+2.66 cm, the v15 guidance-induced contact regression
is largely fixed). Moving correct GT-part recall on `final` ckpt is `0.1990`
(highest non-oracle in the project). Same-part local error 53.49 / 53.23 /
52.91 cm across bc/bv/final vs v15's 55.09 / 59.92 / 54.24. Still ~8-13 cm
short of the v14 K=16 distance oracle (17.60 cm) and v14 K=64 alignment oracle
(40.30 cm local error). Decision-gate verdict: mirror-doubling is worth keeping
but is not a breakthrough; per restart prompt rule we now move to a different
mechanism, not another data/loss-weight knob. See
`analyses/2026-05-01_per_step_guidance_design.md`.

⚠️ v16 wandb diagnostic: train-time `contact_alignment_contact_score`,
`contact_composite_contact_score`, and `contact_alignment_moving_same_part_recall`
are degenerate-early-maximum selectors (peak at epochs 5/5/35 respectively
with very small values). The ckpt-of-record uses `contact_mean_min_dist`
which is fine; do not promote any of the three to ship-metric without
adding a sanity floor (e.g. require `contact_mean_min_dist < X` first).

2026-05-01 v17-C result: largest single-sample contact gain in project
history, on the v16 `best_contact.pt` ckpt without retraining. Per-step
decoded-geometric guidance only (no post-hoc):
`full_guided` contact `21.77 cm` (v16 raw 26.79; v16 `full_guided` 28.91;
v14 K=16 composite oracle `17.94 cm`). Moving coupled `0.3428` (v16 raw
0.2734; v14 K=64 alignment oracle 0.3339 — **v17-C single-sample beats the
K=64 oracle on coupling**). Moving IoU `0.4388` (v14 K=16 0.4472).
Moving correct GT-part recall `0.2020` (v14 K=16 0.2378). Same-part
object-local position error `46.13 cm` (v14 K=16 oracle 46.32 cm —
**v17-C single-sample matches the K=16 composite oracle on local error**).
Design success threshold: 2 of 3 pass (contact ≤ 22.60 cm: pass at 21.77;
local error ≤ 48 cm: pass at 46.13; correct-part ≥ 0.22: miss at 0.2020,
−1.8 pp). Per-step inner loop flips 60.67% of base tokens vs naive
baseline on average, much deeper than the 0–30% flip rate of post-hoc-only
guidance. Detail: `analyses/2026-05-01_v17_per_step_result.md`.

Decision-rule outcome (per design doc §4): "v17-C clearly beats v17-B" →
proceed to v17-D (stacked per-step + post-hoc) and v17-E (per-step budget
sweep at iters ∈ {20, 50}).

2026-05-01 v17-D + v17-E sweep result: v17-E budget scales monotonically;
v17-D (stacked per-step + post-hoc) is *worse* than v17-C, so MaskControl's
canonical stack does not stack on PIANO's deeper RVQ stack. v17-E.50
single-sample beats every K-oracle baseline:

| variant | contact | coupled | IoU | correct-part | local-err |
|---|---:|---:|---:|---:|---:|
| v17-D stacked (10 + 30) | 22.91 | 0.3283 | 0.4380 | 0.1961 | 47.93 |
| v17-E.20 (per-step 20 only) | 18.62 | 0.3559 | 0.4727 | 0.2639 | 42.09 |
| v17-E.50 (per-step 50 only) | 16.50 | 0.3533 | 0.5038 | 0.2746 | 39.02 |

v17-E.50 contact `16.50 cm` < GT VQ roundtrip `18.47 cm` is suspicious for
metric gaming. User visual review confirms v17-E.50 visibly better than
v17-E.20 / v16 raw at contact placement, but body parts are still at the
wrong patch on the object surface (錯位). Detail:
`analyses/2026-05-01_v17_per_step_result.md` (v17-D/E summary section).

2026-05-01 v17 follow-up — γ_int audit + Gumbel addition (v17-F):
- D-A audit of `gamma_int_abs_mean` from v14/v15/v16 wandb shows final
  γ_int ≈ **0.02** (zero-init grew to 0.02 over 80 epochs). ControlNet-style
  gates typically grow to 0.5–1.0 → IntXAttn is gated **1/25 of typical**.
  Indicates Stage B is heavily **under-using** the structured z_int input;
  the v9–v16 training-time decoded contact loss was helping via the direct
  gradient on decoded motion, not via amplifying the cross-attention gate.
  Architectural lever (re-init γ_int, hard-bypass channel for
  contact_target_xyz) deferred until v17-F decides whether inference TTT is
  enough.
- MaskControl uses pretrained MoMask VQ + frozen base + train only the
  control adapter with pure CE → **MoMask codebook is not the bottleneck**.
  GT VQ roundtrip 18.47 cm is "vanilla MoMask training-objective
  reconstruction quality", not a codebook capacity ceiling.
- Route 1 implementation landed: Gumbel-Softmax / Concrete relaxation in
  the per-step inner loop (matches MaskControl's `each_iter` block,
  source-verified diff). New `--per-step-gumbel-scale` CLI (default 1.0 =
  MaskControl-equivalent; 0.0 = pre-v17-F PIANO behaviour, back-compat).
  Detail: `analyses/2026-05-01_v17_diagnostics_and_gumbel.md`.

2026-05-01 v17-F sweep result (Gumbel **negative** on PIANO):

| condition | contact | coupled | IoU | correct-part | local |
|---|---:|---:|---:|---:|---:|
| v17-C-ng (10, OFF, sanity) | 21.80 | 0.3422 | 0.439 | 0.206 | 45.94 |
| v17-F.10 (10, **ON**)      | 23.53 | 0.3251 | 0.403 | 0.177 | 49.02 |
| v17-E.20-ng (20, OFF, sanity) | 18.19 | 0.3550 | 0.475 | 0.271 | 41.90 |
| v17-F.20 (20, **ON**)         | 19.36 | 0.3196 | 0.472 | 0.219 | 42.79 |

Gumbel-OFF reruns sanity-match originals (v17-C / v17-E.20). Gumbel-ON
regresses every metric at both budgets (correct-part −2.9 / −5.2 pp,
contact +1.2 / +1.7 cm). Most likely root cause: PIANO's frozen baseline
residual_emb_sum dominates the decode embedding magnitude, so adding
Gumbel noise to the small base contribution makes inner-loop gradients
noisy across iters; AdamW (β=0.5) doesn't average it out. MaskControl
ignores residual entirely during per-iter so the same noise injection
is well-conditioned in their setting. Same multi-quantizer-residual
incompatibility that killed v17-D (post-hoc stacking).

**Decision**: do not ship Gumbel on PIANO. Default
`--per-step-gumbel-scale=0.0`. Keep flag for any single-quantizer reuse.
Ship configs unchanged: **v17-E.20** (contact 18.62 / IoU 0.473 /
correct-part 0.264 / local 42.09; ~80 min wallclock) or **v17-E.50**
(contact 16.50 / IoU 0.504 / correct-part 0.275 / local 39.02;
~140 min wallclock with metric-gaming caveat).

**Inference path is now near-saturated**: post-hoc stacking, Gumbel
noise both regress; budget sweep at diminishing returns. Remaining
lever per D-A is the **architectural γ_int gate** (final value 0.02,
~1/25 of ControlNet typical). Next branch is P1 inference-time γ_int
boost ablation (`gamma_int_boost ∈ {1, 2, 5, 10, 20}` on top of
v17-E.20 base config). Detail:
`analyses/2026-05-01_v17f_gumbel_result_and_p1_plan.md`.

2026-05-01 v17-G result (γ_int inference boost — **negative**):

| boost | raw cont | per-step cont | raw IoU | per-step correct | per-step local |
|---:|---:|---:|---:|---:|---:|
| 1  | 26.79 | 18.67 | 0.382 | 0.267 | 42.31 |
| 2  | 25.99 | 19.93 | **0.425** | 0.275 | 46.99 |
| 5  | **126.20** | **82.32** | 0.202 | 0.058 | 136.48 |
| 10 | 167.83 | 110.58 | 0.115 | 0.034 | 169.92 |
| 20 | 164.03 | 109.78 | 0.106 | 0.035 | 169.03 |

Sanity (b1) reproduces v17-E.20 within RNG noise. boost=2 is
mixed (raw IoU +4.3 pp, but raw correct-part −2.5 pp; per-step every
metric flat-to-worse). **boost ≥ 5 catastrophic** — model goes OOD,
contact > 100 cm, correct-part < 0.05. Confirms `swap` column
monotonically blowing up (z_int amplification mechanism IS plumbed
correctly): the issue is the rest of the trained MaskTransformer is
calibrated to γ_int ≈ 0.02 and can't tolerate inference-time boost.

**v17 inference-side TTT path SATURATED.** Five distinct levers
tested: per-step (positive), post-hoc stacking (negative), Gumbel
(negative), residual context (deferred), γ_int boost (negative).
Ship configs unchanged: **v17-E.20** or **v17-E.50** (with
metric-gaming caveat).

Diagnosis refinement: γ_int is *undertrained*, not under-applied. The
v9–v16 training loss didn't push γ_int above 0.02 because zero-init +
8-layer-deep gradient dilution + CE/contact-aux objectives that don't
directly reward gate growth all point to a slow plateau. **Next branch
is P2 — re-init γ_int at positive constant + finetune Stage B from
v16 ckpt** (first training-side experiment after 6 weeks of inference
work). Sweep candidates γ_init ∈ {0.1, 0.5, 1.0}, finetune 5–10
epochs, ~3 h per candidate × 3 candidates ≈ 9 h server time. Detail:
`analyses/2026-05-01_v17g_gamma_int_boost_result.md`.

2026-05-01 source-level re-diagnosis (post-v17-G close-out): a
`/restart` Phase-2 trace through the actual repo code (not from prior
synthesis docs) found the "v17 inference-side path SATURATED" claim
was incomplete. Two un-tested inference-side levers exist:

1. **`final.pt` was never evaluated with v17-E recipe.** All v17 work
   used v16 `best_contact.pt`. Per PROGRESS row above, v16 `final.pt`
   has `correct-part recall 0.199` (highest non-oracle in project) and
   `same-part local 52.91 cm` — both strictly better than `best_contact.pt`'s
   0.176 / 53.49 cm. Best_contact is selected on the legacy `mean_min_dist`
   only (`train_generator.py:1257-1260`), which is exactly the
   "checkpoint-selection metric ≠ ship metric" trap.
2. **Inference per-step loss is a strict subset of training loss.**
   Training-time `_target_trajectory_loss_canonical` includes
   `part_margin_weight` (wrong-part margin) and `segment_consistency_weight`
   (object-local offset stability) and `moving_frame_extra_weight=2.0`.
   Inference per-step `_masked_contact_l2` has none — just per-part L2
   to target_world. The visual "right area, wrong patch" failure is
   directly explainable by missing `part_margin`.

2026-05-03 **γ_int re-evaluation** (per user pushback on prior "1/25 of
ControlNet" framing). Source-level + quantitative re-analysis:

- **Full v4–v16 γ_int trajectory**: all 12 runs converge to 0.017–0.036
  range. v05 (160 epochs, 2 × longer) reaches 0.036 — γ_int slowly grows
  past 80 epochs but linear extrapolation suggests γ ≈ 0.05 needs ~480
  total epochs, γ ≈ 0.10 needs ~1100 epochs. NOT a hard plateau.
- **z_int alignment contribution measured directly** by running
  `measure_contact_alignment.py` on full / text_only / swap conditions
  of v16bc raw output (γ_int = 0.02 frozen):
  - text_only (no z_int): correct-part 0.110
  - swap (wrong z_int): correct-part 0.091
  - full (correct z_int): correct-part 0.176
  - codec floor (perfect z_int): correct-part 0.393
  - **z_int at γ=0.02 captures only 23 % of z_int-attributable headroom
    (0.066 / 0.283)**. Substantial but sub-optimal.
- **Architecture comparison**: ControlNet doesn't have a directly-
  comparable scalar γ; the right anchor is LLaMA-Adapter (per-layer
  scalar gating cross-attention output). PIANO at 0.02 vs LLaMA-Adapter
  0.5–1.0 ratio holds (~1/25), but for **architectural reasons**
  (gradient path 8 layers deep, training data 100 × smaller, lower-rank
  conditioning signal) — NOT because PIANO is broken.
- **Revised P2 verdict**: γ_int = 0.02 is below the asymptote but the
  asymptote likely sits at ~0.05–0.10 under PIANO's setup, not
  ControlNet/LLaMA's 0.5–1.0. **Realistic P2 upside: +3–6 pp correct-
  part on guided** (from 0.292 toward 0.32–0.35), NOT closing the entire
  codec floor gap. P2 stays in queue, but candidates revised to
  **{0.05, 0.10, 0.20}** (incremental ramp) and **0.5/1.0 EXCLUDED**.

N1 visual review rendered: `runs/visualizations/stageB_v17E50_final_review/`,
`stageB_v17E20_final_review/`, `stageB_v17E50_bc_review/` — same 10 clips
per run for side-by-side comparison.

Detail: [analyses/2026-05-03_gamma_int_re_evaluation.md](analyses/2026-05-03_gamma_int_re_evaluation.md).

2026-05-03 **Unified metric overhaul + training-vs-inference bottleneck
diagnosis**. Per user's metric review (2026-05-02), implemented N1/N2
penetration (22-joint sphere vs object PC convex hull, body-level only;
finger-level explicitly out of scope per InterDiff/CHOIS/HOI-Diff
convention), N3/N8 weighted_local_error with miss penalty, N6 soft IoU
±2 frame, N7 mean_jerk + KS distance to GT, N4 codec-floor-normalized
% absorbed. Ran full set on GT_orig + GT_roundtrip + 22 v17 conditions.

Diagnosis (per user's request, training-vs-inference attribution):

**Training is the dominant bottleneck.** Decomposition of correct-part
recall headroom (model raw 0.199 → guided 0.292 → codec floor 0.393):

- training-side delta (best_contact → final.pt raw): +2.3 pp (10.6 % of headroom)
- inference-side per-step contribution: +9.3 pp (47.9 % of headroom)
- **52 % of correct-part headroom remains uncaptured**

Training improvements translate ~1:1 into guided gains AND don't pay
plausibility tax. Inference per-step pays:
- mean_jerk **8 × GT_orig** (291 vs 36 m/s³) — independent of budget/Gumbel/boost
- mean_pen **+0.4 cm vs GT_orig** (1.66 vs 1.25)
- frac_pen_gt_2cm **+13 pp vs GT_orig** (54 vs 41 %)

v17-E.50 + final.pt has 4 independent metric-gaming flags
(mean_min_dist 16.86 < codec floor 18.47; pen +0.4 cm; pen-2cm frac
+13 pp; jerk 8×). Should NOT ship as default. Conservative ship config:
**v17-E.20 + final.pt** (cont 19.69 > codec floor, pen 1.53 acceptable,
corPt 0.241, wLoc 28.93).

Decision tree update:

- **N1 visual review** (block before any E.50 ship)
- **Ship default → v17-E.20 + final.pt** (defensible, code-free)
- **N2 mid-loop residual refresh** (B3') — narrowed expected upside 1–3 pp
- **B4 = P2 with revised γ_init {0.05, 0.1, 0.2}** — first training-side
  experiment; cheapest path to lift raw distribution
- **B6 alignment-aware VQ retrain** — biggest expected upside (codec
  floor is dominant alignment ceiling at 60 pp), but biggest investment
- **B7 OMOMO-style explicit contact_target input** — long-term backup

Detail: [analyses/2026-05-03_unified_metric_results.md](analyses/2026-05-03_unified_metric_results.md).
Reproducer: `python scripts/stage_b_generator/summarize_unified_metrics.py`.

2026-05-02 **VQ codec floor on alignment metrics measured for the first
time** — paradigm shift: previously-unmeasured codec floor reveals all
v17 inference results are MUCH closer to the achievable ceiling than
the raw numbers suggested. Trigger: user asked "are GT/GT_roundtrip
references still trustworthy?". Answer: numbers correct, but reference
set was incomplete — `GT_roundtrip vs GT_orig` on alignment metrics had
never been measured (only `GT_orig vs GT_orig` self-check, which is
trivial 1.0/0.0).

| metric | prior reference | **codec floor** | v17-E.50+final.pt | gap to floor |
|---|---|---:|---:|---:|
| moving correct-part recall | (1.0 implied) | **0.393** | 0.292 | 0.101 (74% absorbed) |
| moving same-part local error | (0.0 implied) | **28.61 cm** | 36.11 cm | 7.5 cm (much smaller) |
| moving contact IoU | (1.0 implied) | **0.640** | 0.507 | 0.133 |
| mean_min_dist | 18.47 cm (codec) | 18.47 cm | **16.86 cm** | **−1.61 cm = GAMED** |

**Major implications**:

1. v17-E.50 + final.pt mean_min_dist 16.86 cm < codec floor 18.47 cm is
   **direct evidence of metric gaming** (no model can physically beat
   the codec floor on its own GT input). Penetration metric needed
   before E.50 can ship.
2. Inference-side ceiling much closer than thought: v17-E.50 + final.pt
   has absorbed ~74% of available correct-part headroom. B3' residual
   refresh realistic upside drops from "many pp" to "1–3 pp".
3. **Prior "VQ codec not the bottleneck" claim narrowed**: it's true on
   `mean_min_dist`, but VQ codec is now the **dominant** bottleneck on
   alignment metrics (60 pp of correct-part recall lost to codec alone).
   B6 (alignment-aware VQ retrain) becomes a real candidate after
   inference-side exhausted.
4. **Recommended ship change**: default to v17-E.20 + final.pt
   (mean_min_dist 19.69 cm > codec floor — physically defensible),
   not v17-E.50. E.50 stays as opt-in with metric-gaming caveat
   pending penetration metric (T1.1) and visual review (N1).

Detail: [analyses/2026-05-02_codec_floor_baselines.md](analyses/2026-05-02_codec_floor_baselines.md).
Reproducer: same `measure_contact_alignment.py` with
`generated_dir=gt_roundtrip`, `gt_dir=gt_original`. Stored at
`runs/eval/stageB_codec_floor_alignment/summary.json`.

2026-05-02 B1 + B2 + B3 server results synced (3 branches from
`analyses/2026-05-01_v17_re_diagnosis.md` decision tree):

- **B1 (final.pt re-eval) — partial WIN at high budget**:
  v17-E.50 + final.pt → contact 16.86 cm / IoU 0.507 / **correct-part
  0.292 (project SOTA on raw single-sample)** / **same-part local
  36.11 cm (project SOTA)**. Beats prior v17-E.50 + best_contact on
  every alignment metric. v17-E.20 + final.pt regresses (correct-part
  0.241 vs prior 0.264) — final.pt's wider raw distribution requires
  per_step ≥ 50 to translate into a guided-side win.
- **B2 (part_margin + segment_consistency) — NEGATIVE**: every variant
  with weight > 0 regresses correct-part recall vs sanity (pm=0)
  rerun. pm=1.0 alone: 0.198 (−6.7 pp). pm=1.0 + sc∈{0.1,0.5,1.0} only
  partially recovers. Sanity rerun matches prior v17-E.20 within RNG
  (0.265 vs 0.264) → new code path is correctly wired; the aux terms
  themselves do not transfer to inference.
- **B3 (residual drift) — confirms ceiling**: even sanity (pm=0) has
  mean |drift| 5.93 cm; max 49.86 cm per clip. drift scales with
  part_margin weight (pm=2.0 → 11.86 cm) — explains B2 failure: aux
  terms make base-token flipping more aggressive, which amplifies
  divergence between the per-step optimiser's frozen-baseline residual
  context and the actual post-residual-rerun residuals.

**New ship config**: `v17-E.50 + final.pt` (subject to visual review).
Metric-gaming caveat (contact 16.86 < GT VQ roundtrip 18.47) carries
over from prior E.50, but correct-part 0.292 is project-best.

**Next branch**: implement B3' — mid-loop residual refresh
(`per_step_residual_refresh_every=N`) before any P2 commitment. Drift
> 5 cm is the actual inference ceiling; B2 is expected to flip from
negative to positive once drift < 2 cm. P2 (γ_int re-init) continues
to wait.

Detail: [analyses/2026-05-02_v17h_results.md](analyses/2026-05-02_v17h_results.md).
Reproducer: `python scripts/stage_b_generator/summarize_v17h_results.py`.

2026-05-01 B2 + B3 implementation landed (this commit):

- `src/piano/inference/contact_guidance.py`: new
  `_per_step_target_loss_with_aux` helper; inner-loop loss site +
  `guide_with_contact` accept `per_step_part_margin_weight`,
  `per_step_part_margin_m`, `per_step_segment_consistency_weight`
  (default 0.0 = back-compat, equal to legacy `_masked_contact_l2`).
  Aux terms run in **object-local frame** (matches training site at
  `decoded_contact_loss.py::_target_trajectory_loss_canonical`).
- B3 residual drift diagnostic: `guide_with_contact` now re-evaluates
  the same loss on the FINAL post-residual-rerun motion and records
  `loss_opt_last_inner`, `loss_final_post_residual`, `residual_drift`
  inside `info["per_step"]`. Auto-writes to `guidance_trace.json` via
  the existing `per_clip_guided[i]["full"]["guidance_info"]` dump, so
  every B1/B2 run also produces B3 data.
- `qual_eval.py`: new `--per-step-part-margin-weight` /
  `--per-step-part-margin-m` /
  `--per-step-segment-consistency-weight` CLI; written to
  `guidance_trace.json` top-level for run-attribution.
- `run_v13_target_trajectory.sh` / `run_v17_per_step_guidance.sh`:
  pass-through env vars `PER_STEP_PART_MARGIN_WEIGHT`,
  `PER_STEP_PART_MARGIN_M`, `PER_STEP_SEGMENT_CONSISTENCY_WEIGHT`.
- `tests/test_contact_guidance_per_step.py`: 5 new tests covering
  fast-path back-compat, wrong-part penalty correctness, segment-drift
  correctness, object-frame translation invariance, gradient flow.
  All 15 (15 new + 10 prior) per-step tests + 21 sibling
  contact_guidance tests pass locally.

Refines (does not refute) the P2 hypothesis: γ_int growth from v15
(0.0161) → v16 (0.0204) was +27 % but contact only improved 3 %
(27.62 → 26.79 cm). Diminishing returns at the trained plateau. v17-G
boost = 2 was already mixed → network's tolerance window for γ_int
change is < 2× at inference. P2 with γ_init=0.5 is asking for a 25×
change post-training; lower-risk γ_init ∈ {0.05, 0.1, 0.2} is
preferable. Revised decision tree B1–B5 with B1 (re-eval final.pt,
zero code, ~3.5 h) and B2 (port part_margin to per-step, ~50 LOC,
~4 h) ahead of any P2 commitment. Detail:
`analyses/2026-05-01_v17_re_diagnosis.md`.

## Stage Status

Stage 1 pseudo-labels:

- Current InterAct label track: v11.
- Important fields: contact body part, closest-surface target xyz,
  3-class phase, support state, object pose in canonical/body frame.
- Durable doc: `analyses/pseudo_label_pipeline.md`.

Stage A Interaction Predictor:

- Shipped state: v6.
- Predictor of record: server checkpoint `runs/training/predictor/final.pt`
  (local sync may only include eval JSONs).
- Durable doc: `analyses/stageA_design.md`.
- Revisit only if downstream diagnostics show Stage A labels/predictions are
  the limiting factor.

Stage B Motion Generator:

- Active training-loss bottleneck: the generator must learn temporally bound
  manipulation, not only spatial proximity.
- Current implementation includes residual `z_int` conditioning and decoded
  contact auxiliary loss through full RVQ prediction.
- v13 replaced the old arbitrary min-distance decoded loss with a part-specific
  object-local contact-target trajectory objective plus a moving-object
  local-velocity term.
- v14 keeps the target-trajectory objective but uses all-mask MaskGIT/CFG
  first-step logits, straight-through Gumbel hard codebook lookups, and full
  residual RVQ rollout for the decoded auxiliary path.
- v15 adds alignment-aware negatives/segment binding to the same decoded path,
  and evaluates full-RVQ target guidance as a sampling-time correction.
- v16 keeps v15's objective and turns on deterministic mirror duplication for
  training only, matching MoMask/HumanML3D's mirrored-data assumption more
  closely than the old stochastic v0.7 `mirror_prob=0.5` test.
- Main training script: `src/piano/training/train_generator.py`.
- Main model wrapper: `src/piano/models/motion_generator.py`.
- Decoded contact loss: `src/piano/training/decoded_contact_loss.py`.
- Current training runner:
  `scripts/stage_b_generator/run_v16_alignment_mirror.sh`.
- Current no-retrain diagnostic runner:
  `scripts/stage_b_generator/k_sample_oracle.py`.
- Durable doc: `analyses/stageB_compact.md`.

Stage C Joint Finetune:

- Not started.
- Do not start until the Stage B reranking/guidance baseline passes metric and
  visual review.

## Stage B Evidence Timeline

| Step | Result | Decision |
|---|---|---|
| v0.1 | initial `z_int` adapter, weak visual effect | needed object pose |
| v0.2 | object pose added, token signal up but body mostly in place | inspect MoMask path |
| v0.3/v0.4 | MoMask encoder normalization bug fixed | first major correction |
| B0 | contact metric introduced; v0.4 full 20.86 vs GT roundtrip 11.29 on canonical 5 | measure contact, not mp4 vibes |
| v0.5 | lower CE but worse contact | CE is misaligned |
| v0.6 | per-head gamma improved canonical 5 to 16.03 | useful but limited |
| v0.7 | mirror augmentation regressed to 29.50 | data symmetry not the fix |
| v0.3-delta | trainable-copy variant regressed after dead-init fix | architecture swap exhausted |
| B1 | contact checkpointing exposed CE/contact decoupling | useful diagnostic, not enough |
| B3 | inference guidance mixed wins/losses | base logits unstable |
| C1/v0.8 | residual `z_int` alone: 43.62 full | not enough |
| C2/v0.9 | decoded contact aux: 29.19 full on 20 clips | real control signal |
| C2b/v0.10 | full-RVQ path: 25.27 full on 20 clips | partial gain |
| v0.11 | diagnostics showed weight 0.10 had small gradient share | motivated sweep |
| v0.12 | weights 0.20/0.30/0.50/0.80 all near 32 cm on 80 clips | stop blind weight sweeps |
| K=16 oracle | best-of-K 17.93 cm on 80 clips | reranking/guidance becomes main path |
| K=16 visual review | body is near object but weakly synchronized to object motion | distance-only reranking is insufficient |
| temporal coupling metric | moving coupled frame frac 0.323 | optimize/rerank for coupling, not only distance |
| K=16 composite oracle | coupled frac 0.351, contact 18.08 cm | only modest gain; K=16 pool lacks enough coupled samples |
| v13 target trajectory loss | best_val 31.57 cm, coupled 0.265 | soft target improves internally but hard sampled output stays on v12 line |
| v13 RVQ diagnostics | soft_train_full 14.78 cm / 0.443 coupled; mixed_pred_all 33.50 cm | soft-hard and base-token path gaps are real |
| v14 sampled-ST loss | best_contact 27.37 cm, coupled 0.277 | first single-sample contact gain, but temporal binding remains weak |
| v14 wandb history | best_contact selected at epoch 65: train-time 26.33 cm / coupled 0.308 | decoded aux optimized, contact/coupling remain stochastic |
| v14 K=16 oracle | distance 16.80 cm; composite 17.17 cm, remeasured 17.94 cm / coupled 0.3715 | v14 candidate pool improves; selection/guidance is now the main lever |
| v14 RVQ diagnostics | mixed_pred_all 29.31 cm vs v13 33.50; pred base + GT residual 29.81 vs 35.92 | sampled/base path improved, residual bottleneck remains |
| v14 contact alignment | composite moving IoU 0.447; correct GT-part recall 0.238; local part error 46 cm | spatial contact metrics are being gamed; use part/target-aware guidance |
| v14 K64 alignment oracle | remeasured 18.71 cm; coupled 0.334; moving IoU 0.452; correct GT-part recall 0.250; local part error 40 cm | alignment selection gives only modest local-position gain; K64 pool lacks enough aligned samples |
| v15 alignment-guided | best_contact full 27.62 cm; guided 31.57 cm; moving IoU 0.380 and correct GT-part recall 0.168 | negative/neutral; alignment losses did not create GT-quality samples |
| v16 mirror-doubled | deterministic original+mirror train-set duplication implemented and tested locally | next server run; tests MoMask/HumanML3D data assumption without conflating with old stochastic mirror p=0.5 |
| v16 server result | bc full 26.79 cm; bv full 28.13; final full 28.05; final correct GT-part recall 0.1990; same-part local 52.91-53.49 cm | partial positive vs v15; still 8-13 cm short of v14 K-oracle baselines; decision-gate triggers next-mechanism branch (v17 per-step) |
| v17 per-step guidance | inference-time MaskControl-style each_iter logit optimisation, runs on existing v14/v15/v16 ckpts unchanged; v17-C runner + tests landed | next server run; ablation plan in analyses/2026-05-01_per_step_guidance_design.md (v17-A baseline, v17-B post-hoc only, v17-C per-step only, v17-D stacked, v17-E iter sweep) |
| v17-C result | full_guided 21.77 cm contact / 0.3428 coupled / 0.4388 IoU / 0.2020 correct-part / 46.13 cm same-part local | matches v14 K=16 composite oracle on local error; beats K=64 alignment oracle on coupling; advance to v17-D + v17-E |

## v0.12 Details

Runner:

```bash
bash scripts/stage_b_generator/run_v12_contact_weight_sweep.sh
```

Default weights:

```text
decoded_contact_aux.weight = 0.20, 0.30, 0.50, 0.80
```

Best 80-clip result by full contact:

| Rank | checkpoint | full |
|---:|---|---:|
| 1 | w02 best_val | 31.82 |
| 2 | w08 final | 32.17 |
| 3 | w03 best_contact | 32.39 |
| 4 | w02 final | 32.51 |

Gradient diagnostic:

| weight | decoded grad median | final decoded loss | best full |
|---:|---:|---:|---:|
| 0.20 | 5.57% | 0.1558 | 31.82 |
| 0.30 | 7.97% | 0.1340 | 32.39 |
| 0.50 | 13.02% | 0.1128 | 32.87 |
| 0.80 | 19.27% | 0.0988 | 32.17 |

Conclusion: decoded-contact surrogate optimization is working mechanically, but
it does not translate monotonically to sample-time contact.

K-sample oracle on v12 w02 best_val:

| metric | value |
|---|---:|
| single-sample mean | 32.22 cm |
| K=16 sample mean | 31.64 cm |
| K=16 best-of-K mean | 17.93 cm |
| K=16 best-of-K median | 14.50 cm |
| best under 22 cm | 70% |
| best under 25 cm | 80% |

Saved best samples re-measured with `measure_contact_distance.py` at
`18.70 cm`, close to the oracle score. This confirms the selected saved output
is in the GT VQ roundtrip band.

Per-subset K=16 best-of-K:

| subset | single | best-of-K |
|---|---:|---:|
| chairs | 18.51 | 8.44 |
| imhd | 42.90 | 29.38 |
| neuraldome | 37.87 | 21.66 |
| omomo_correct_v2 | 29.60 | 12.23 |

IMHD remains the hardest subset; the worst outlier is
`20230901_wangwzh_suitcase_suitcase_lefthand_carry_3_0` at `116.76 cm` even
after K=16 reranking.

Visual review of the K=16 best samples: the positive signal is real
object-position conditioning. The body is usually near the object and often
oriented/moving in the same broad direction. The failure is stronger: body
motion is often only colocated with the object trajectory, not temporally bound
to it. In object-moving clips, the person may perform a plausible action near
the object while the object's move timing does not match and no stable contact
is visible.

New diagnostic:

```bash
python scripts/stage_b_generator/measure_temporal_coupling.py \
  --input-dir runs/eval/stageB_v0_12_w02_bv_k16_oracle/best \
  --output-dir runs/eval/stageB_v0_12_w02_bv_k16_oracle/temporal_coupling
```

Result on K=16 distance-reranked best:

| metric | value |
|---|---:|
| ordinary mean contact distance | 0.187 m |
| moving-object frame fraction | 0.555 |
| moving frames with any close tracked body part | 0.475 |
| moving frames with kinematic coupling | 0.323 |
| moving frames close but uncoupled | 0.245 |

Subset coupling:

| subset | moving coupled frame frac |
|---|---:|
| chairs | 0.665 |
| imhd | 0.134 |
| neuraldome | 0.277 |
| omomo_correct_v2 | 0.379 |

This supports the user's visual assessment: distance-only contact is not a
strong enough proxy for "the person is actually manipulating the object."

Composite K=16 reranking result:

| metric | distance K=16 | composite K=16 |
|---|---:|---:|
| contact mean | 17.93 cm | 18.08 cm |
| contact median | 14.50 cm | 14.74 cm |
| under 22 cm | 70% | 70% |
| under 25 cm | 80% | 80% |
| moving coupled frame frac | 0.323 | 0.351 |
| close but uncoupled moving frac | 0.245 | 0.222 |

Composite reranking changed only `12/80` selected samples. Among changed clips,
average contact distance worsened by only `0.99 cm`, while moving-coupled frame
fraction improved by `0.554`. The aggregate gain is small because most clips
kept the same sample and IMHD has weak coupled candidates in the K=16 pool.

Offline rerank over the stored K=16 candidate scores shows the ceiling:

| selection rule | contact mean | moving coupled frac |
|---|---:|---:|
| distance-only | 17.93 cm | 0.325 |
| current composite | 18.08 cm | 0.354 |
| high coupling weight ~1.0 | 19.36 cm | 0.386 |
| max-coupled oracle | 20.67 cm | 0.390 |

Per-subset max-coupled capacity within K=16:

| subset | max-coupled mean | contact at max-coupled | clips with any >=0.5 |
|---|---:|---:|---:|
| chairs | 0.838 | 6.99 cm | 3/3 moving clips |
| imhd | 0.180 | 37.53 cm | 2/20 |
| neuraldome | 0.368 | 27.91 cm | 5/17 |
| omomo_correct_v2 | 0.456 | 18.27 cm | 7/20 |

Conclusion: tuning rerank weights is not enough. The model distribution needs
stronger temporal-binding generation or training; IMHD/baseball/suitcase cases
are the clearest blockers.

Subset decomposition for v12 w02 best_val:

| subset | GT orig | GT roundtrip | full | codebook gap | model gap |
|---|---:|---:|---:|---:|---:|
| chairs | 12.04 | 12.09 | 19.07 | 0.05 | 6.98 |
| imhd | 8.41 | 22.55 | 42.30 | 14.13 | 19.75 |
| neuraldome | 14.80 | 19.30 | 33.63 | 4.50 | 14.33 |
| omomo_correct_v2 | 17.11 | 19.95 | 32.27 | 2.84 | 12.32 |

The old "codebook is negligible" conclusion only held on the canonical 5 clips.
On 80 clips, IMHD has a large roundtrip/codebook issue.

## Next Work

Immediate (revised after 2026-05-01 source-level re-diagnosis;
supersedes the prior P2-first plan):

1. **B1 — Re-eval v17-E.20 + v17-E.50 on v16 `final.pt`** (zero code,
   ~3.5 h server). All v17 work was on `best_contact.pt`; `final.pt`
   has +2.3 pp correct-part / −0.58 cm same-part local advantage that
   has not yet been combined with per-step. Override
   `SOURCE_RUN_DIR=runs/training/generator_v16_alignment_mirror CKPTS=final`
   in `run_v17_per_step_guidance.sh`. If correct-part > 0.22 and local
   < 42 cm without metric gaming → new ship config.
2. **B2 — Port `part_margin_weight` and `segment_consistency_weight`
   from training to per-step inner loss** (~50 LOC,
   `contact_guidance.py::_generate_with_per_step_guidance`, no
   retraining; ~4 h server per ablation). Visual "right area, wrong
   patch" failure is directly explained by these terms being absent at
   inference. Sweep `part_margin_weight ∈ {0.5, 1.0, 2.0}`. If
   correct-part jumps ≥ 5 pp → ship `v17-H` as new default.
3. **B3 — Diagnose residual-context drift** (~30 LOC,
   `guide_with_contact`, ~4 h server). Per-step inner loop uses a
   frozen `baseline_residual_emb_sum`; final motion uses
   post-residual-rerun residuals. If |L_final − L_opt| > 5 cm,
   per-step is being misled and a mid-loop residual-context refresh
   becomes worth the compute.
4. **B4 — P2 (γ_int re-init + Stage B finetune)** with revised γ_init
   candidates. Existing P2 plan's γ_init=0.5/1.0 is too aggressive
   given v17-G's "boost ≥ 5 catastrophic" measurement (network's
   tolerance window < 2× at inference). Revised candidates:
   `{0.05, 0.1, 0.2}`. Run only after B1+B2+B3 don't close the
   correct-part gap. ~1 day code + ~12 h server.
5. **B5 — Pivot to OMOMO-style explicit `contact_target` input**.
   Final fallback if B1–B4 fail. Architecture change using existing
   Stage A predictor output channel.

Ship configs frozen until B1/B2 land: **v17-E.20** (recommended
default; contact 18.62 / correct-part 0.264 / local 42.09 cm) or
**v17-E.50** (best metrics with metric-gaming caveat per visual
review).

Decision tree detail:
`analyses/2026-05-01_v17_re_diagnosis.md` §5.

Secondary diagnostics:

- Subset-specific codebook audit if IMHD remains poor after v14 K-sample search.
- Full visual review of v16 best_contact hard cases before treating any contact
  gain as semantically meaningful.

## Environment

Server:

```bash
cd /media/gpu-server-1/4TB_for_data/Cai/PIANO/PIANO
conda activate piano
```

Local workspace:

```text
e:\Project\2026-04-13
```

Local tests should use the `piano` conda environment when possible.
