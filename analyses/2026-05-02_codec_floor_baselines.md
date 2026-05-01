# 2026-05-02 — VQ codec floor on alignment metrics (paradigm shift)

Triggered by the user question "current GT / GT_roundtrip references —
are they still trustworthy? do we need new metrics?"

Local check: rerun `measure_contact_alignment.py` with
`generated_dir = gt_roundtrip` and `gt_dir = gt_original` (using the
existing 80-clip GT data — **zero new compute beyond what was already
saved during every eval**). This measures **what the MoMask VQ codec
itself does to the alignment metrics** when the ONLY change between
"generated" and "reference" is a clean GT motion → VQ encode → VQ decode.

The result is a **previously-unmeasured codec floor** — the absolute
ceiling no inference-side or training-side intervention on the existing
MoMask backbone can exceed.

## What was wrong with the prior reference set

| Reference baseline | Status before today |
|---|---|
| GT_orig `mean_min_dist` (13.09 cm) | ✓ measured every eval |
| GT_roundtrip `mean_min_dist` (18.47 cm) | ✓ measured every eval |
| GT_orig vs GT_orig on alignment | trivial self-check (everything = 1.0 / 0.0) |
| **GT_roundtrip vs GT_orig on alignment** | **NEVER MEASURED** |

The naming `<EVAL_PREFIX>_alignment_to_gt_roundtrip` referred to
"generated motion vs GT_roundtrip"; we never inserted GT_orig as the
generated stand-in to measure the codec floor on alignment.

## Codec floor numbers (80-clip matched eval)

Run: `runs/eval/stageB_codec_floor_alignment/summary.json`. Reproducer:

```bash
python scripts/stage_b_generator/measure_contact_alignment.py \
  --generated-dir runs/eval/<EVAL_PREFIX>_gt_roundtrip_80/gt_roundtrip \
  --gt-dir       runs/eval/<EVAL_PREFIX>_gt_roundtrip_80/gt_original \
  --output-dir   runs/eval/stageB_codec_floor_alignment
```

| metric | GT self-check | **Codec floor** (GT_rt vs GT_orig) | v17-E.50 + final.pt | gap to floor |
|---|---:|---:|---:|---:|
| `moving_contact_temporal_iou` | 1.000 | **0.640** | 0.507 | 0.133 |
| `moving_contact_recall_on_gt` | 1.000 | 0.653 | — | — |
| `right_part_contact_recall_on_gt` (all frames) | 1.000 | 0.551 | — | — |
| **`moving_right_part_contact_recall_on_gt`** | 1.000 | **0.393** | **0.292** | **0.101** |
| `moving_same_gt_part_contact_recall_on_gt` | 1.000 | 0.534 | — | — |
| `same_gt_part_local_position_error_m` | 0.000 | 21.14 cm | — | — |
| **`moving_same_gt_part_local_position_error_m`** | 0.000 | **28.61 cm** | **36.11 cm** | **7.5 cm** |
| `moving_target_part_local_error_m` | 0.000 | 31.01 cm | 43.48 cm | 12.5 cm |
| `mean_min_dist_per_frame` | 0 | 18.47 cm | 16.86 cm | **−1.61 cm (gamed)** |

## Interpretation: the codec floor is much lower than "1.0 / 0.0" implies

The codec floor on **moving correct-part recall is 0.393, not 1.0**. So
even a perfect generator that exactly reproduced GT motion through the
MoMask VQ → VQ pipeline could only achieve correct-part recall 0.393 on
this 80-clip eval. The remaining 60.7 % of GT moving contact frames have
their part-membership flipped by VQ codec noise alone.

Mechanism: contact judgement is a **threshold on a sigmoid of distance**
(`contact_score = max(distance_sigmoid(d, threshold_per_part, sigma=0.03), kinematic_score) >= 0.5`).
The VQ encode/decode introduces ~5 cm L2 noise in joint positions
(consistent with the 13.09 → 18.47 cm increase in `mean_min_dist`).
That 5 cm noise lives directly inside the 3 cm sigmoid transition width
of the contact threshold — so frames sitting near the contact boundary
flip from "in contact" to "not in contact" (or vice versa), and the
nearest-part identity flips when two parts are within 5 cm of each other
near the surface (e.g. left vs right hand on a table).

Practical implication: **on this metric definition, no model running on
the existing MoMask backbone can exceed 0.393 moving correct-part recall.**

## Reinterpretation of v17-E.50 + final.pt (project SOTA)

Before today's measurement:

> v17-E.50 + final.pt: correct-part 0.292. "Project best on raw single-sample,
> still 70 pp short of the perfect-1.0 ceiling — large remaining headroom."

After today's measurement:

> v17-E.50 + final.pt: correct-part 0.292 / **codec floor 0.393** = **74 % of
> the codec ceiling already absorbed**. Remaining inference-side headroom
> is ~10 pp at the very most. B3' (residual refresh) realistic upside:
> 1–3 pp. Major future gains require either changing the codec or
> changing the metric definition (looser contact threshold).

Same applies to local error:

| | gap (raw) | gap relative to codec floor |
|---|---:|---:|
| same-part local error | 36.11 vs perfect 0 → 36 cm gap | 36.11 vs floor 28.61 → **7.5 cm gap** |
| target-part local error | 43.48 vs perfect 0 → 43.48 cm gap | 43.48 vs floor 31.01 → 12.47 cm gap |

The "real" model gap is **about a quarter** of what the raw numbers
suggested.

## What this also tells us about `mean_min_dist`

v17-E.50 + final.pt mean_min_dist = 16.86 cm < codec floor 18.47 cm.

Codec floor 18.47 cm is the value GT motion **itself** achieves after
encode/decode. So **any model output strictly below 18.47 cm on
`mean_min_dist` is, by definition, doing something the codec floor of
real human motion cannot do**. There are only two possibilities:

1. **Metric gaming**: optimizer pushes some body part to artificially close
   to some object PC sample, exploiting the `min over 5 parts of min over
   PC` reduction.
2. **Penetration**: body parts pass through object surface (negative SDF
   not penalised by `mean_min_dist`).

Either way, `mean_min_dist < 18.47 cm` is **not a positive signal**. It's
direct evidence that the metric is being optimised against, not the
underlying physical objective.

The ship-config recommendation must therefore change:

- **v17-E.20 + final.pt** (mean_min_dist 19.69 cm > codec floor — physically
  defensible): defensible ship.
- **v17-E.50 + final.pt** (mean_min_dist 16.86 cm < codec floor — gamed):
  ship ONLY if visual review confirms physical plausibility AND
  pene­tration metric (T1.1, not yet implemented) is non-negative.
  Otherwise step back to E.20.

## Implications for next-branch priority

The re-diagnosis decision tree was:

| | priority | predicted gain | confidence |
|---|---|---|---|
| N1 visual review of v17-E.50 + final.pt | first | — | high |
| N2 mid-loop residual refresh (B3') | second | ~5 pp correct-part | medium |
| N3 P2 γ_int re-init + finetune | last | 1–10 pp | low |

Updated, given codec floor:

- **N1 unchanged** (still required to confirm v17-E.50 isn't gamed).
- **N2 expected gain narrowed**: previously thought "many pp" possible;
  now ceiling is correct-part 0.393, so realistic max upside 0.10 pp.
  Still worth doing — but ROI assessment changes; if N2 only buys 1–2 pp
  it may not be worth the dev cost vs other directions.
- **N3 (P2) headroom revisited**: training-side change can't help on
  metric definitions — codec floor is the same. Any gain has to come
  from either (a) actually beating the codec by training a better VQ
  (deferred per "VQ not the bottleneck" — needs revisiting on alignment
  metrics specifically, see new branch below), or (b) closing the small
  remaining gap to the codec floor.
- **NEW: revisit "VQ codec not the bottleneck" claim**. The prior
  conclusion (in `analyses/2026-05-01_v17_diagnostics_and_gumbel.md`)
  was based on `mean_min_dist`. On alignment metrics — which are the
  current ship metrics — VQ codec **IS** the dominant remaining
  bottleneck. Codec floor 0.393 vs perfect 1.0 = 60 pp lost to codec
  alone.

Possible new branch B6: **train a higher-fidelity RVQ** specifically
optimised for contact-relevant joints (hands + feet + pelvis), not
generic motion reconstruction. Cost: significant (retrain VQ-VAE).
Justification only if N1+N2 confirm we are genuinely at the codec
floor on alignment.

## What metrics need to change (revised from yesterday's metric review)

The metric review (`analyses/...` user conversation 2026-05-02) listed
T1–T3 priority补强. After today's codec floor measurement, the priority
order shifts:

| | reason | priority |
|---|---|---|
| **T1.5 — make codec floor a permanent reference (every eval)** | new finding; trivial cost | **highest** |
| T1.1 — penetration / SDF metric | catches `mean_min_dist < codec_floor` gaming directly; without it, can't ship E.50 | very high |
| T1.2 — weighted local error (with miss penalty) | unchanged; correct gap interpretation | high |
| T1.3 — subset-stratified ship gate | unchanged | medium |
| T1.4 — use GT_orig as the "good" baseline, not GT_roundtrip | partially obsoleted by today's measurement: **codec floor IS the proper baseline**, not GT_orig | medium |
| T2.x | unchanged | medium / low |

## Action items

### Immediate (zero new compute, just re-summarise)

1. **Add codec floor to `summarize_v17h_results.py` output** so every
   future ablation table compares "model − codec_floor" not "model".
2. **Reinterpret all historical v14/v15/v16/v17 numbers** with codec
   floor as the reference. Most of the field has been compressed into
   one decimal place worth of room above codec floor.

### Short-term

3. Implement **T1.1 penetration metric** before shipping any E.50
   variant. `mean_min_dist < codec_floor` without penetration check is
   ungroundable.
4. **Change ship-config default to v17-E.20 + final.pt** (correct-part
   0.241 / mean_min_dist 19.69 cm > codec floor / local 43.35 cm).
   Trade: lower correct-part recall but no metric-gaming red flag.
   E.50 stays as opt-in with caveats.

### Medium-term

5. Consider **looser contact_threshold** in metric (e.g. 0.3 instead of
   0.5). Will raise codec floor and make all downstream numbers more
   meaningful, at the cost of comparability with prior numbers.
6. **B6: revisit VQ as bottleneck on alignment metrics.** Possibly
   train an alignment-aware VQ-VAE — but only after exhausting
   inference-side moves.

## References

- Reproducer one-liner above.
- Codec floor summary: `runs/eval/stageB_codec_floor_alignment/summary.json`.
- Alignment metric definitions:
  `scripts/stage_b_generator/measure_contact_alignment.py`,
  `src/piano/data/pseudo_labels/extract_contact.py::ContactConfig`
  (per-part `distance_thresholds` + sigmoid `distance_sigma=0.03`).
- Prior assertion that VQ codec is not the bottleneck (now narrowed to
  `mean_min_dist` only):
  `analyses/2026-05-01_v17_diagnostics_and_gumbel.md` §"MaskControl
  source-verified ... VQ codebook is not the bottleneck".
- Yesterday's full metric review (in conversation): T1.x / T2.x / T3.x
  priorities — superseded above.
- This re-diagnosis: `analyses/2026-05-01_v17_re_diagnosis.md`,
  result doc `analyses/2026-05-02_v17h_results.md`.
