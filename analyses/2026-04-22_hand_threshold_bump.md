# Hand Threshold 0.08 → 0.12 — 2026-04-22

## Context

v6 `summary.json` aggregates showed hand contact under-firing on every
subset, even the ones whose aggregate-level pass bars had been met:

| subset | hand seq_reached (v6, 0.08) | zero-contact seq |
|---|---:|---:|
| chairs | L 61% / R 59% | 2.73% |
| imhd | L 63% / R 58% | 13.5% (flagged) |
| neuraldome | L 38% / R 40% | 49.0% (flagged) |
| omomo_correct_v2 | L 58% / R 63% | 23.1% (flagged) |

The low chairs zero-contact rate (2.73%) had masked the hand issue on
that subset — pelvis carries the "any body part" signal, so the
subset-level flag does not fire. But chairs has 39-41% of sequences
where no hand contact is ever registered, which directly starves
Stage A's hand-contact supervision even on the cleanest subset.

The "fix only if v3 imhd zero-contact > 20%" gate from the earlier
Beyond-P0 list (2026-04-21 §c #1) was too coarse: it tracked
subset-level zero-contact, which conflates all 5 body parts, not
per-part seq_reached.

## Evidence — threshold sweep

`runs/threshold_sweep/2026-04-20_193818/<subset>/analysis.md` already
had the per-body-part sweep done at v1 time. Pulling seq_reached at
candidate thresholds:

**left_hand / right_hand seq_reached:**

| subset | 0.08 (v6) | 0.10 | **0.12** | 0.14 |
|---|---:|---:|---:|---:|
| chairs L  | 61.2% | 70.0% | **74.8%** | 78.8% |
| chairs R  | 59.3% | 68.0% | **73.3%** | 77.6% |
| imhd L    | 63.0% | 69.9% | **74.5%** | 77.2% |
| imhd R    | 58.3% | 65.4% | **68.6%** | 70.1% |
| neuraldome L | 38.0% | 42.5% | **47.4%** | 51.4% |
| neuraldome R | 39.6% | 45.3% | **50.0%** | 54.5% |
| omomo L   | 58.1% | 68.7% | **74.1%** | 75.5% |
| omomo R   | 62.6% | 73.7% | **79.0%** | 79.8% |

**Raw hand-to-object distance distribution:**

| subset | left_hand p25 | p50 |
|---|---:|---:|
| imhd | 0.074 | 0.157 |
| neuraldome | 0.151 | 0.314 |

0.08 catches only the p20-p30 slice on imhd and well below p15 on
neuraldome. Medians at 0.16 m (imhd) and 0.31 m (neuraldome) reflect
that neuraldome has many large objects (`bigsofa`, `box`, `desk`)
whose surfaces are naturally 15-25 cm from the wrist during normal
interaction — arms wrap around rather than fingertips-only.

## Rationale for 0.12 over 0.10 or 0.14

**Anatomy**: SMPL wrist joint is 5-8 cm inside the forearm; palm
surface is 8-10 cm out from that joint. For a gripping pose where the
hand wraps a handle / bat shaft / box edge / pan rim, the wrist ends
up 10-15 cm from the closest mesh surface. 0.12 m is right inside this
band.

**Curve elbow**: imhd / chairs / omomo all plateau near 0.12-0.14. Going
from 0.12 to 0.14 buys only 3-4 pp more seq_reached but starts
capturing "hand near but not touching" poses — riskier FP trajectory.

**Persistence filters defend against FP**: the 0.12 threshold is the
sigmoid midpoint, and the resulting soft contact score must then
survive `median_filter_size=5` temporal smoothing and `min_contact_duration=3`
minimum-length filtering in `_filter_short_contacts`. A hand that
briefly passes within 12 cm of an object without actually touching it
does not produce 3+ consecutive frames of high contact score.

**0.10 leaves too much on the table**: it still under-covers
neuraldome (only 42-45% seq_reached vs 47-50% at 0.12), and the
4-5 pp extra from 0.12 on imhd / omomo is exactly the "gripping pose"
class we care about.

## What this does NOT fix

Some v5/v6 false-zero-contact sequences are mesh-layer issues that no
hand threshold can repair:

- **`imhd/suitcase_lefthand_push`**: the suitcase mesh may omit the
  pull handle, so the "handle" the user is pressing has no geometry
  for the distance query to hit.
- **Handle-type objects in omomo** (`suitcase`, `vacuum`, `mop`): same
  concern. `vacuum`/`mop` are already skipped in preprocessing.

These need a `mesh.bounds` vs `object_pc.bounds` comparison per object
to flag incomplete meshes, separate from thresholds. Tracked in
PLAN §3.1.

## Expected v7 deltas

| subset | metric | v6 | v7 (expected) |
|---|---|---:|---:|
| chairs | hand seq_reached L/R | 61/59% | ~75/73% |
| chairs | zero-contact seq | 2.7% | ~1.5% (pelvis already carries most) |
| imhd | hand seq_reached L/R | 63/58% | ~75/69% |
| imhd | zero-contact seq | 13.5% | ~8% (quality flag may stop firing) |
| neuraldome | hand seq_reached L/R | 38/40% | ~47/50% |
| neuraldome | zero-contact seq | 49.0% | ~35% (flag will still fire; this subset has a residual data-side issue) |
| omomo | hand seq_reached L/R | 58/63% | ~74/79% |
| omomo | zero-contact seq | 23.1% | ~15% |

Contact frame_rate per hand will roughly double on most subsets (the
sweep shows 0.08→0.12 raises frame_rate from 18-30% to 29-54%
depending on subset). Phase and target outputs change too — more
frames classified as `stable-contact` / `manipulation`, target
assignments now populated on those frames. Support is unaffected
(it only reads pelvis + hand binary contact, not distance).

## Downstream risk assessment

- **target patch entropy**: more contact frames → more patch assignments
  with soft weights. Entropy mean should stay > 1.2 (new frames use the
  same sigma=0.12 Gaussian kernel that already gave entropy 1.21-1.79 in
  v6).
- **phase `stable-contact` fraction**: will rise on chairs and imhd
  (more is_contact frames that are stationary). Not a correctness
  problem — these are real contact frames that v6 missed.
- **phase `manipulation` fraction**: similarly rises on imhd
  (held-bat-rotating now also registers hand contact, so rotation-while-
  contact enters manipulation instead of falling through to approach).
- **HMM refinement**: emission means are re-fit per sequence from
  heuristic labels with frozen EM, so distribution shift in the
  heuristic output is absorbed. No action needed.

## Action Items (→ PLAN.md)

- [x] Bump `DEFAULT_DISTANCE_THRESHOLDS["left_hand"/"right_hand"]` from
      0.08 to 0.12; update module docstring to reference this analysis.
- [x] 16/16 regression tests pass (tests operate on synthetic contact
      arrays, not distance thresholds — no test changes needed).
- [ ] v7 rerun on server (same `rerun_pseudo_labels_interact.sh`, ~5 h).
- [ ] Compare v7 aggregates — target: imhd zero-contact below 10%,
      neuraldome below 40%, all hand seq_reached ≥ 47% (neuraldome
      residual attributable to data-layer mesh issues).
- [ ] Confirm per-hand frame_rate roughly tracks the sweep curve
      prediction (verifies the fix landed as intended).
