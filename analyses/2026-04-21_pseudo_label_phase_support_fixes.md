# Pseudo-Label Phase/Support Design Fixes — 2026-04-21

## Context

While v2 extraction was running on the server, a full review of
`src/piano/data/pseudo_labels/` turned up four design-level bugs that
have nothing to do with v2's threshold choices. They don't show up in
the aggregate-stats histograms (the numbers look plausible), but they
would surface in visualization or in Stage A held-out accuracy.
Waiting for v2 to finish before fixing them means rerunning v3 just to
get trustworthy labels, so the fixes landed locally — with regression
tests — before v2 completed.

## Four P0 design bugs

### 1. Phase `is_contact` looked at hand contact only

[extract_phase.py, pre-fix line 108-109]:
```python
hand_contact = np.maximum(contact_state[:, 0], contact_state[:, 1])
is_contact = hand_contact > config.hand_contact_threshold
```

The `stable-contact` / `manipulation` branches only fire when
`is_contact` is True.

**Consequence**: in a chair-sitting sequence (pelvis on chair, hands
idle) `hand_contact = 0` → `is_contact = False` → the entire seated
stretch falls through to `approach` / `pre-contact`. PLAN.md §1.2's
pass bar `chairs manipulation > 30%` was meaningless under the old
definition — the real "stable contact" frames for chairs are
pelvis-chair, which phase simply could not see.

**Fix**: `is_contact = contact_state.max(axis=-1) > contact_threshold`.

### 2. `obj_vel` used translational velocity only

[extract_phase.py, pre-fix line 111-116]:
```python
obj_vel[1:] = np.linalg.norm(
    np.diff(object_positions, axis=0), axis=-1
) * config.fps
```

Compared against `object_velocity_eps = 0.02 m/s`.

**Consequence**: an in-place bat swing or chair rotation leaves
`object_positions` roughly static → `obj_vel ≈ 0` → frame goes to
`stable-contact` instead of `manipulation`. InterAct distribution
makes this common: chairs has 60 objects with many "rotate the chair
in place" clips; imhd's baseball bat / tennis racket / golf club
sequences are characteristically in-place rotations around the grip;
neuraldome's badminton/baseball clips follow the same pattern.

**Fix**: add angular velocity. `object_rotations` is already saved in
`motions/<seq>.npz` (preprocess_interact.py:241); run_all.py simply
wasn't plumbing it through.
```python
ang_vel[1:] = ||diff(object_rotations)|| * fps
is_moving = (trans_vel > eps_t) | (ang_vel > eps_r)    # eps_r = 0.3 rad/s
```
Axis-angle finite differences are a good approximation at per-frame
Δt (20 fps → Δangle << 0.5 rad).

### 3. HMM EM was allowed to update parameters → state-id drift

[refine_phase_hmm.py, pre-fix]:
```python
hmm = GaussianHMM(..., init_params="")
# All params set manually; state k's means are aligned to phase k
hmm.fit(features)                    # default params="stmc" → M-step updates means/covars/transmat
refined = hmm.predict(features)      # returns raw state ids
```

`hmm.fit`'s default `params="stmc"` updates start, transmat, means,
and covars every EM iteration. On short or noisy clips EM can drift
`means[0]` out of the "approach" cluster and into another one —
state 0 then corresponds to a different phase semantically. `predict`
happily returns ids that no longer match the phase constants.

**Consequence**: silent bug — the aggregate histogram still looks
normal, but a label written as `phase=MANIPULATION` could actually
correspond to a different cluster. Only visualization would catch it.

**Fix**: `GaussianHMM(..., params="", init_params="")` freezes every
parameter. `fit()` then runs forward-backward without any M-step
updates, and `predict()` performs Viterbi over the fixed parameters.
State k stays bound to phase constant k by construction. `n_iter` is
lowered to 1 — extra E-steps are wasted work under a frozen model.

### 4. Support used a median filter on categorical ids

[extract_support.py, pre-fix line 91]:
```python
support = median_filter(support, size=config.median_filter_size)
```

`support ∈ {0=both_feet, 1=single_foot, 2=sitting, 3=hand_support}`
are **unordered** categories.
`median([single_foot=1, sitting=2, hand_support=3]) = 2 (sitting)` is
a semantically meaningless "in-between" value — there is no ordering
among the three ids to begin with.

**Fix**: a hand-written `_majority_filter` using
`np.bincount(window).argmax()`, with edge-replication padding (same as
`scipy.ndimage` `mode="edge"`).

## Code changes (local, pending commit)

| File | Change |
|---|---|
| `src/piano/data/pseudo_labels/extract_phase.py` | `is_contact` over any body part; add `object_rotations` parameter + angular-velocity term; config rename `hand_contact_threshold`→`contact_threshold`, `object_velocity_eps`→`translational_velocity_eps`; new `rotational_velocity_eps=0.3` |
| `src/piano/data/pseudo_labels/refine_phase_hmm.py` | `GaussianHMM(params="")` freezes M-step; `HMMConfig.n_iter=1`; `build_phase_features` takes `object_rotations` and emits `(T, 4)` features with angular velocity |
| `src/piano/data/pseudo_labels/extract_support.py` | Drop `median_filter`; add `_majority_filter`; rename `SupportConfig.median_filter_size`→`smoothing_window` |
| `src/piano/data/pseudo_labels/run_all.py` | Plumb `object_rotations` through to `extract_interaction_phase` and `build_phase_features` |
| `tests/test_pseudo_labels.py` | New — 7 regression tests, one minimal synthetic sequence per fix |

Grep after the rename confirmed no other call sites referenced the
old field names — no regression surface.

## Verification

A local conda env `piano` (python 3.10 + rtree from conda-forge +
numpy / scipy / scikit-learn / hmmlearn / pytest, with the package
installed editable via `pip install -e . --no-deps`) runs pytest:

```
tests/test_pseudo_labels.py::test_phase_sitting_enters_stable_contact PASSED
tests/test_pseudo_labels.py::test_phase_rotation_only_enters_manipulation PASSED
tests/test_pseudo_labels.py::test_phase_rotation_only_is_stable_without_rotation_signal PASSED
tests/test_pseudo_labels.py::test_hmm_state_ids_preserve_phase_semantics PASSED
tests/test_pseudo_labels.py::test_build_phase_features_shape_with_rotation PASSED
tests/test_pseudo_labels.py::test_support_majority_filter_no_ordinal_artifacts PASSED
tests/test_pseudo_labels.py::test_support_extraction_sitting_sequence PASSED
============================== 7 passed in 6.88s ==============================
```

Each test reproduces the minimal buggy scenario (sitting only,
rotation only, HMM state drift, majority window with three different
categories) rather than asserting on arbitrary values.

## Trust levels for v2 stats

v2 was run with the **old** code, so its `summary.json` fields split
into trustworthy and not:

| Metric | v2 trustworthy? | Why |
|---|---|---|
| Contact rate per body part | ✓ | Determined by threshold + geometry; orthogonal to these four fixes |
| Zero-contact seq fraction | ✓ | Same |
| Target entropy / patch coverage | ✓ | Target extraction path untouched |
| Geometric sanity (hand-to-obj-center dist) | ✓ | Preprocess-layer signal |
| Phase `manipulation reached` | ✗ | Polluted by #1 + #2 |
| Phase `stable-contact frame rate` | ✗ | Same |
| HMM-refined phase histogram | ✗ | #3 state drift |
| Support `sitting frame rate` | △ (partial) | Core logic unchanged, but the median filter injects spurious labels at transition frames |

v2 judgement plan: read the pass bar off contact and target only;
defer phase and support to v3.

## Implications

- v3 is unavoidable regardless of v2's outcome — trustworthy phase /
  support labels require the new code.
- v2 still carries information: contact and target tell us whether
  the thresholds are calibrated. If either fails its pass bar,
  ContactConfig / TargetConfig need more tuning before v3 starts.
- If v2 contact and target pass, commit the batch and start v3
  (~5 h) to get final phase / support labels.

## Post-visualization validation (2026-04-21 PM)

After pulling v2's `summary.json` locally, `visualize_finished_subsets.sh`
was run on all 4 subsets. 14 representative clips were spot-checked by
hand (chairs 3 + imhd 3 + neuraldome 5 + omomo 3). Findings split
into three groups.

### (a) The four P0 fixes were directly validated

| Sequence | v2 mislabel | Evidence for which fix |
|---|---|---|
| `chairs/Sub0012_Obj116_Seg0_105` (81 frames) | All 81 frames actually seated, but v2 phase 100% approach | P0-1 (`is_contact` hand-only → pelvis-only sitting drops to approach) |
| `chairs/Sub0001_Obj116_Seg0_0` (199 frames) | sitting 81% but approach 72% | P0-1 |
| `chairs/Sub0005_Obj116_Seg0_360` (127 frames) | Seated with right hand on chair; the 25 frames where the hand briefly left the chair were all labelled approach | P0-1 (pelvis contact should keep the frame in stable during hand dropout) |
| `imhd/songzn_bat_righthand_swing_4_0` | User reports "rotating the bat continuously"; v2 put 47/80 frames in approach + stable-contact | P0-1 + P0-2 in compound |
| `neuraldome/subject02_bigsofa_0` | Later stretch fully seated on sofa with hand resting on it, but sitting 55% vs approach 70% | P0-1 |

### (b) A fifth P0-class bug surfaced — `support=sitting` false positive

This was not part of the original four fixes. Two neuraldome clips
exposed it:

- `subject01_bigsofa_330` (206 frames) — v2 labelled sitting 96%;
  user report: **person stands in front and pushes the sofa the
  entire time, never sits**.
- `subject01_chair_0` (332 frames) — v2 labelled sitting 63%; user
  report: **person stands behind the chair and pushes/pulls it
  repeatedly, never sits**.

Root cause: when pushing or dragging a large object, the pelvis joint
is often within 20 cm of the mesh surface (backrest to waist, large
sofa armrest to belly). v2's `support_state` logic was
`if pelvis[t]: SITTING` without any disambiguator.

**Impact on v2 stats**: the chairs `support sitting = 64%` headline
number probably contains 30-40% false positives (standing-and-pushing
counted as sitting). Stage A's support supervision would be directly
contaminated.

**Fix (applied 2026-04-21 PM, two conjunctive gates)**:

*Gate 1 — pelvis XZ velocity*: a 1 s moving average of XZ-plane
pelvis speed > 0.15 m/s vetoes sitting. Reference scales: pushing a
sofa 0.2-0.5 m/s, pushing a chair 0.3-0.5 m/s, walking > 1 m/s;
seated small shifts stay < 0.10 m/s. This gate rejects the
**moving-while-pushing** case.

*Gate 2 — object geometrically below pelvis* (user-proposed,
physically principled): the geometric signature of sitting is that
the seat surface is directly below the pelvis (the seat takes the
body's weight). The direction from pelvis to the mesh's closest
point must have a Y-component < -0.3 (at least 30% downward, angle
with -Y within ~72°). This rejects the
**standing-beside-object-stationary** case — when a person stands
next to a chair's backrest, the pelvis joint is horizontally within
threshold but the closest-point direction is sideways, not down.

Both gates are AND-combined — either failing vetoes sitting:
- `bigsofa_330` (user walks in front pushing): velocity gate fires
  (pelvis walks while pushing) → rejected.
- `chair_0` (user stands behind pulling): velocity gate AND below
  gate fire (when standing, the closest point is horizontally at the
  backrest, not under the pelvis) → rejected.
- Hypothetical "stationary beside chair with pelvis against
  backrest": velocity gate passes, below gate fires → rejected.
- True sit (pelvis over the seat): both gates pass → sitting.

Implementation: `extract_support_state` gains `joints` /
`object_mesh` / `object_positions` / `object_rotations` parameters.
`SupportConfig` gains `fps`, `sitting_max_pelvis_horz_speed = 0.15`,
`sitting_velocity_window_sec = 1.0`, and
`sitting_min_downward_component = 0.3`. `run_all.py` threads mesh
and transforms into `extract_support_state`.

Four new tests, one per gate dimension:
- `test_support_push_object_not_classified_as_sitting` (velocity
  gate: 1 m/s horizontal pelvis → not sitting)
- `test_support_stationary_sitting_still_classified_as_sitting`
  (velocity gate sanity: stationary pelvis → sitting)
- `test_support_rejects_sitting_when_pelvis_beside_object` (below
  gate: pelvis flush against a box's side face → not sitting)
- `test_support_allows_sitting_when_object_below_pelvis` (below gate
  sanity: pelvis above a box's top face → sitting)

### (c) Beyond-P0 — deferred to PLAN §3.1

Observed but not urgent:

1. **Hand threshold 0.08 m too strict for irregular / bulky objects**.
   Four sequences were all-zero hand contact despite genuine contact:
   holding a bat by the thick end (`bat_holdhead_hit_0_1501`),
   pushing a suitcase (`suitcase_lefthand_push_0_0`), carrying a
   large box with outstretched arms (`neuraldome/box_1565`), holding
   a pan (`neuraldome/pan_360`). Same pattern across four clips.
   Candidate fixes: relax the threshold to 0.10-0.12; add elbow /
   palm joints to the tracked set; or use a palm or finger proxy for
   the closest-point query. Gate: v3 imhd zero-contact > 20%.

2. **InterAct suitcase mesh may omit the handle**. User flagged that
   the rendered point cloud might not include the pull handle. If
   the mesh lacks the handle, the suitcase push sequences are
   unrecoverable at the pseudo-label layer. This is a data-layer
   issue, not fixable in the extraction code. Gate: compare
   `mesh.bounds` against `object_pc.bounds` Y extent on one suitcase
   seq; if the mesh is 50 cm short, confirmed.

3. **P0-2's impact is narrower than expected**. omomo
   `whitechair_001` ("grab chair from behind → rotate through a
   turn → place in front") was labelled 76% manipulation by v2 —
   because "carry object while turning" produces translation of both
   body and object, triggering `obj_vel`. Pure in-place rotation
   (bat swing) is the real P0-2 scenario. imhd is affected most;
   other subsets see marginal benefit.

## Action Items (→ PLAN.md)

- [x] Code fixes applied locally + tests green (2026-04-21)
- [x] `piano` conda env established (minimal for pseudo-label tests)
- [x] v2 extraction completed → summary.json judged against
      contact / target pass bar (chairs / imhd / omomo contact pass,
      neuraldome 49% over bar; target 4/4 fail)
- [x] Visualization for all 4 subsets + 14 clips spot-checked by hand
- [x] Additional P0 fix (sitting dual gate, §b above); tests still green
- [x] Target sigma 0.05 → 0.12, calibrated against v2 entropy numbers
- [x] HMM NaN fallback — v2 had 5/8475 "startprob_ must sum to 1
      (got nan)" exceptions; the refine step now returns heuristic
      labels on failure instead of dropping the sequence
- [ ] Commit all local fixes → start v3 rerun (same command, ~5 h)
- [ ] v3 summary.json → phase / support pass bar (now meaningful)
- [ ] If v3 chairs `sitting > 25%` still fails → check whether the
      pelvis-velocity gate is rejecting valid sits (threshold 0.15
      may need to be loosened to 0.20)
- [ ] If v3 imhd zero-contact is still > 20% → revisit the hand
      threshold / tracked joints (Beyond-P0 #1 above)
- [ ] Update `configs/training/predictor.yaml` support_weight to
      0.1 until v3 validates support labels
