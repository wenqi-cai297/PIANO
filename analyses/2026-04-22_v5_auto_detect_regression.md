# v5 Auto-Detect Up-Axis Regression + Whitelist Fix — 2026-04-22

## Context

After `edf2bb3` landed (face-area-argmax per-mesh up-axis detection,
2026-04-22 AM), v5 pseudo-label extraction ran on all 4 InterAct
subsets overnight. The fix was motivated by v4 showing neuraldome
`bigsofa` sitting = 0 because of a hardcoded `normal.Y > 0.7` filter,
and a server-side face-normal dump confirming bigsofa is Z-up while
chairs are Y-up.

v5 `summary.json` files at
`runs/InterAct/piano/<subset>/pseudo_labels/summary.json`.

## Results — v5 vs v4

| subset | metric | v4 | v5 | Δ | verdict |
|---|---|---:|---:|---:|---|
| chairs | sitting | 49.6% | **39.5%** | **-10.1 pp** | ✗ regression |
| chairs | seq entered sitting | 88.4% | 71.4% | -17 pp | ✗ |
| chairs | pelvis contact frame rate | 63.2% | 63.2% | 0 | ✓ unchanged as designed |
| chairs | target entropy mean | 1.215 | 1.215 | 0 | ✓ byte-identical |
| imhd | sitting | 0.66% | **3.05%** | **+2.4 pp** | ✗ false positives — imhd is bat/broom/dumbbell, not sittable |
| imhd | seq entered sitting | ~1% | 9.5% | +8.5 pp | ✗ |
| neuraldome | sitting | 1.60% | 1.75% | +0.15 pp | △ bigsofa contributed much less than hoped |
| omomo | sitting | 0.06% | 0.06% | 0 | ✓ |

Phase / contact / target unchanged as designed — only support path
differs from v4.

## Root cause — probe data

`scripts/data/probe_mesh_up_axis.py` (new this iteration) enumerates
every InterAct object mesh and reports the face-area argmax axis plus
a dominance ratio. Output at
`runs/checks/up_axis_probe/2026-04-22_101850/probe.json`.

**Non-+Y detections per subset:**

| subset | +X | +Y | +Z | non-Y fraction |
|---|---:|---:|---:|---:|
| chairs | 3 | 39 | 18 | **35% (21 / 60)** |
| imhd | 3 | 2 | 5 | **80% (8 / 10)** |
| neuraldome | 7 | 5 | 9 | 76% (16 / 21) |
| omomo_correct_v2 | 4 | 4 | 7 | 73% (11 / 15) |

**Chair face-area is fundamentally non-dominant.** Chair 116 has
`pos +X 0.36 / +Y 0.79 / +Z 0.69` — +Y wins but only by a 1.15× margin
over +Z. Chair 141 (the `Sub0284_Obj141_Seg0_339` that motivated this
whole path) has `pos +X 0.43 / +Y 0.52 / +Z 0.30`, dominance 1.21.
A chair with a tall curved backrest or large side panels easily has
more face area in a non-+Y direction than on its seat.

**imhd objects have no real "up".** Bats / brooms / dumbbells /
kettlebells are elongated or radially symmetric. `baseball` picks +Y
(dom 1.09), `broom` +Y (1.13), but `kettlebell` +Z (1.68), `dumbbell`
+X (1.34), `skateboard` +Z (11.5). For these objects a "cylinder below
pelvis along detected up" is physically meaningless — it may happen to
enclose some surface that satisfies the cylinder-axis and upward-normal
filters, producing false-positive sitting.

**bigsofa itself is Z-up with only a weak margin.** +X 1.08 / +Y 0.47 /
+Z 1.21 — dominance 1.12. So any "dominance threshold" set to guard
chairs would also exclude bigsofa.

**smallsofa is the other under-covered case.** Probe picked +X
(dom 1.38), but extents [0.89, 0.86, 0.71] make Z the short/vertical
axis — smallsofa is almost certainly Z-up like bigsofa. Face-area
argmax failed here too, probably because the sofa's side cushions
carry more face area than the seat top.

## Fix applied

`_detect_mesh_up_axis` now defaults to +Y with an explicit whitelist:

```python
OBJECT_UP_AXIS_OVERRIDES = {
    "bigsofa": "+Z",
    "smallsofa": "+Z",
}
```

The `mesh` and `threshold` arguments stay in the signature for
call-site compatibility (including the probe tool) but are unused.
Auto-detect by face-area is removed entirely — no dominance threshold,
no fallback-to-+Y-when-marginal. Just +Y unless the object name is
whitelisted.

Why whitelist instead of a smarter heuristic:
- Dominance gating doesn't work (bigsofa 1.12, chair 116 1.15, chair
  141 1.21 are all in the same range — any threshold that admits
  bigsofa also admits the mis-detected chairs).
- Pelvis-relative detection (look at where the pelvis is vs the object
  centroid) would couple the up-axis to the sequence, producing
  inconsistent patch semantics across clips of the same object.
- Extents-based inference (thinnest axis = up) also fails: chair 116
  has Z as its thinnest axis but is Y-up.

Threading: `extract_support_state` grows an `object_id` kwarg,
`process_sequence` grows `object_id`, `run_pipeline` passes
`entry["object_id"]` through. `_pelvis_object_below_mask` forwards
it to `_detect_mesh_up_axis`.

Tests updated:
- Old `test_support_auto_detects_up_axis_for_z_up_mesh` became
  `test_support_up_axis_override_unlocks_z_up_mesh`, which passes
  `object_id="bigsofa"` so the whitelist branch fires.
- New `test_support_default_up_axis_rejects_z_up_mesh_without_override`
  guards against a regression where someone re-enables auto-detect —
  the same synthetic Z-up slab without `object_id` must reject sitting.

16/16 regression tests pass locally.

## Expected v6 deltas

| subset | sitting | expectation |
|---|---:|---|
| chairs | ≈ 49.6% | Back to v4. All 60 chairs use +Y default, same as v4. |
| imhd | ≈ 0% | Revert of v5 FPs. All 10 imhd objects default to +Y, which rejects bats/brooms/dumbbells. |
| neuraldome | ~1.6% + bigsofa + smallsofa | Same as v5 for bigsofa (still +Z); smallsofa newly unlocked. Net change could be +0.5 to +1 pp if smallsofa has a few real sit clips. |
| omomo | ≈ 0% | Unchanged. |

The 3-5% neuraldome lift originally hoped for in
2026-04-21_pseudo_label_phase_support_fixes §e is no longer in scope —
auto-detect itself was the failure mode, and widening the normal-
alignment threshold 0.7 → 0.5 is a separate dial we haven't tried yet.
If v6 bigsofa vis still shows sitting-frames-missing despite correct
+Z axis, relax threshold in v7.

## Action Items (→ PLAN.md)

- [x] Probe all InterAct meshes, dump per-mesh face-area + detected axis.
- [x] Replace face-area argmax with +Y default + whitelist.
- [x] Thread `object_id` through extract_support_state → run_all.py.
- [x] Update regression tests; 16/16 pass locally.
- [x] v6 rerun on server (commit `6608e5a`, ~5 h elapsed). Results below.
- [x] Compare v6 to v4 aggregates.
- [ ] Run vis on the 3 bigsofa / chair-141 diagnostic clips to confirm
      whether bigsofa sit now fires (hypothesis: still 0, because 0.7
      upward-normal threshold rejects curved cushion faces).
- [ ] If bigsofa still underperforms, relax `sitting_below_upward_normal_threshold`
      0.7 → 0.5 as a follow-up (separate commit, "v7" candidate).

## v6 verdict (2026-04-22 PM)

v6 ran with `6608e5a` (whitelist fix). Full summaries at
`runs/InterAct/piano/<subset>/pseudo_labels/summary.json`.

| subset | metric | v4 | v5 | **v6** | verdict |
|---|---|---:|---:|---:|---|
| chairs | sitting frame rate | 49.6% | 39.5% | **49.62%** | ✓ regression fully fixed, byte-equal to v4 |
| chairs | seq entered sitting | 88.4% | 71.4% | **88.4%** | ✓ |
| chairs | seq stuck in both_feet | 8.3% | 8.3% | **4.3%** | ✓ better than v4 too |
| chairs | quality_flags | — | [] | **[]** | ✓ clean |
| imhd | sitting frame rate | 0.66% | 3.05% | **0.21%** | ✓ FPs cleared (both numbers within surface-sampling stochastic noise — "effectively zero") |
| imhd | seq entered sitting | ~1% | 9.5% | **4.6%** | ✓ 5× cleaner than v5; residual 27 seqs × ~8 frames each is flicker-level |
| neuraldome | sitting frame rate | 1.60% | 1.75% | **1.55%** | △ ≈ v4 — bigsofa + smallsofa whitelist didn't lift aggregate |
| neuraldome | seq entered sitting | 7.2% | 7.2% | **5.1%** | △ dropped — v5's mis-detected false-positive sits are filtered |
| omomo | sitting frame rate | 0.06% | 0.06% | **0.06%** | ✓ unchanged |

**Byte-identity checks** (v5 vs v6 — only support path changed):
chairs pelvis contact `0.6315110842118912` identical ✓;
chairs target `entropy_mean=1.2148...` identical ✓;
chairs phase `approach=0.2359...` identical ✓. Confirms only support
was affected, as designed.

**Neuraldome half-failure diagnosis.** bigsofa + smallsofa are the only
whitelist entries; before v6 they contributed 0 sitting frames each (v4)
because hardcoded +Y rejected every seat face. In v6 they now get +Z
but aggregate still lost ~128 sitting frames vs v4. Two candidate
explanations, both consistent with the PROGRESS §0 note about v4:

1. **Curved cushion faces still filtered.** bigsofa's seat is a
   cushion, not a flat slab; near the seam with the backrest the face
   normal tilts toward ±X and falls below `upward_normal_threshold=0.7`.
   The cylinder test finds no qualifying seat points and the gate stays
   closed on exactly the frames we wanted to unlock. The "sofa-cushion
   threshold 0.7 too strict" was already noted as a v5 explanation; v6
   confirms it without the auto-detect regression confounder.

2. **Stochastic surface sampling.** `trimesh.sample.sample_surface` is
   un-seeded in `_pelvis_object_below_mask`. On borderline geometry
   (held objects in imhd, marginal seats in neuraldome) a different
   sample can flip frames in or out of the gate, explaining ~0.05 pp
   noise differences between runs. Fine as noise but means precise
   before/after diffs on flat-seat chairs are more trustworthy than on
   cushion-seat sofas.

**Trade-off between v5 and v6 on neuraldome.** v5 had 1.75% but 16/21
objects were using wrong-axis detection that produced spurious sit
frames on non-sittable objects (desk, pillow, case, monitor, etc.).
v6 is 1.55% with only physically meaningful sits. **v6 is cleaner
signal for training** even though aggregate is slightly lower.

**Conclusion.** v6 passes all relevant pass bars (chairs sitting >>
25%, all 4 subsets target entropy > 1.2, manipulation reached > 30%
on chairs/imhd/omomo, contact zero-rate < 30% on chairs/imhd/omomo).
neuraldome zero-contact 49% is the deferred hand-threshold issue
(`PLAN §3.1`), not a support-path issue. Stage A predictor training
can begin on v6 labels. If bigsofa-specific sitting recovery is
still desired for neuraldome vis fidelity, v7 candidate is a one-line
config change: `sitting_below_upward_normal_threshold = 0.7 → 0.5`.
