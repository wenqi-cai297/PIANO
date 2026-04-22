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
- [ ] v6 rerun on server (same `bash scripts/data/rerun_pseudo_labels_interact.sh`, ~7 h).
- [ ] Compare v6 to v4 aggregates — chairs sitting should return to
      ~49.6%, imhd sitting should drop to ~0%.
- [ ] Run vis on the 3 bigsofa / chair-141 diagnostic clips to confirm
      smallsofa sitting actually fires where expected.
- [ ] If bigsofa still underperforms, relax `sitting_below_upward_normal_threshold`
      0.7 → 0.5 as a follow-up (separate commit).
