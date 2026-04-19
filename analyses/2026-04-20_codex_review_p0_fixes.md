# Codex Review + Pseudo-Label P0 Fixes — 2026-04-20

## Context

Codex produced two review documents for the pseudo-label extraction pipeline:

- [`SUGGESTION.md`](../SUGGESTION.md) — overall pipeline review (7 labelled issues
  across P0/P1/P2, plus training-implication checklist).
- `support.md` — deep critique of the support-state label specifically
  (323 lines, proposes a 3-level hierarchy as the long-term redesign).

Both were read, each claim was cross-checked against the actual code, and
the two P0 items were fixed before trusting any output from the
in-flight extraction. Commit `9d11f1a`.

## Results

### Verification of Codex claims against code

Confirmed against source (line numbers at time of review):

| Claim | Code location | Status |
|---|---|---|
| fps defaults are 30, data is 20 | `extract_contact.py:40`, `extract_phase.py:55`, `refine_phase_hmm.py:116`; `runs/summary.json` target_fps=20; `run_all.py` never overrode | Confirmed |
| Patch ids are not stable | `_farthest_point_sample` started from `np.random.randint(n)` and `cluster_surface_patches` was called inside `extract_contact_target` per sequence | Confirmed |
| Soft-assignment kernel uses `d` not `d²` | `geometry.py:113` `logits = -dists / (2.0 * sigma ** 2)` — linear distance | Confirmed |
| Contact velocity is world-frame | `extract_contact.py:95` uses `compute_joint_velocities(joints)` directly; code comment acknowledges the approximation | Confirmed (severity likely overstated by Codex) |
| `median_filter` on categorical support ids | `extract_support.py:91` applies `median_filter` to `{0,1,2,3}` | Confirmed |
| Default `both_feet` for ambiguous frames | `extract_support.py:86-88` | Confirmed |
| Only 5 tracked joints (wrist/ankle/pelvis) | `BODY_PART_INDICES` | Confirmed |

Nothing in the reviews was found to be off-base. Two P1-ish claims
(world-frame velocity impact, HMM state drift) are real but severity is
not established — they go into the "measure before fixing" queue.

### P0 fixes applied

**1. FPS propagation (`run_all.py`).** Added `_resolve_fps(data_dir,
override)` that reads `target_fps` from `<data_dir>/summary.json`, falls
back to `<data_dir>/../summary.json` (top-level preprocess summary), and
exposes a `--fps` CLI override. The run-time config is now built from
the resolved value:

```python
resolved_fps = _resolve_fps(data_dir, fps)
contact_cfg = ContactConfig(fps=resolved_fps)
phase_cfg   = PhaseConfig(fps=resolved_fps)
```

`preprocess_interact.py` also learned to write `source_fps` and
`target_fps` into each subset's `summary.json`, so future preprocess
outputs are self-describing.

**2. Deterministic per-object patch atlas.** `cluster_surface_patches`
and `_farthest_point_sample` now take a `seed`. `run_all.py` computes
the atlas once per `object_id` using `seed = md5(obj_id)[:8]` and caches
it on disk at `<output_dir>/patch_atlas/<obj_id>.npy`. The atlas is
passed into `extract_contact_target` via a new `patch_centers` argument;
the previous recomputation path is kept as a non-deterministic fallback
only for single-sequence debugging. The same object now yields the same
patch ids across re-runs and machines.

### Rerun mechanics

New `scripts/data/rerun_pseudo_labels_interact.sh`:

1. `tmux kill-session -t piano-labels` (configurable via env var).
2. Move existing `<subset>/pseudo_labels/` to
   `<subset>/pseudo_labels.<ts>_pre_fps_fix/` (backup, not delete).
3. Re-run `piano-pseudo-labels` for all 4 subsets with the fixes in
   place.

The previous in-flight run was producing velocity-inflated labels
(1.5×) and non-deterministic patch ids — not salvageable.

## Observations

- The review was precise: every "P0" claim was directly traceable to a
  code line, not pattern-matched from training-best-practice lists.
  Signal-to-noise was high.
- The `support` label is the weakest component on paper, but the most
  concrete symptoms will only show up in the rerun stats and videos.
  Held off on the hierarchical redesign (`support.md` §recommendation)
  until we see whether sitting/lying sequences are mislabeled in
  practice.
- The P2 soft-assignment kernel bug (`-d` vs `-d²`) is real but low
  impact given `sigma=0.01` already produces near-hard nearest-patch
  behavior. Not fixing pre-rerun.

## Diagnosis

The P0 fps bug was a **silent** bug: all defaults said 30, the
preprocess emitted 20, nothing wired them together, and no test
exercised the velocity threshold at the actual data rate. Adding
`target_fps` to the per-subset summary and reading it back during
extraction converts the implicit assumption into an explicit contract.

The patch-id bug was a **latent** bug: the pipeline ran fine, the
labels trained, but the resulting model would have been unable to
ground class id `3` to a consistent object region across sequences.
This is the kind of issue that only shows up as a mysterious accuracy
ceiling weeks into training.

## Implications

- The rerun output (pending) replaces the first extraction as the
  trustable v1 pseudo-labels.
- Visualization + per-subset stats come next. Without them we can't
  tell whether the P1/P2 items (world-frame velocity, support
  semantics, body-part coverage) actually need code changes or just
  config tweaks (e.g., lower `support_weight` and move on).
- Training should not start until the rerun finishes and its
  per-subset histograms look non-degenerate.

## Action Items (→ PLAN.md)

- [x] Commit `9d11f1a`: fps + patch atlas fixes + rerun script.
- [ ] User runs `scripts/data/rerun_pseudo_labels_interact.sh` on
      server, waits for completion (~1.5h expected).
- [ ] Write pseudo-label stats aggregator: per subset contact-rate per
      body part, phase/support histogram, patch-id entropy, short-contact
      fraction, HMM non-convergence count.
- [ ] Render 5-10 videos per subset with
      `visualize_pseudo_labels.sh` for qualitative spot-checks.
- [ ] Decide on P1/P2 scope (world-frame velocity, support rename,
      median→majority filter, kernel `-d²`, closest-surface-point target,
      joint coverage) **only after** stats + videos are in hand.
