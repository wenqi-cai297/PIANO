# Switched Data Track: InterAct-only, Dropped CHOIS-OMOMO — 2026-04-19

## Context

After successfully preprocessing OMOMO (4919 sequences from CHOIS bundle) and
inspecting InterAct (4 subsets, 8478 sequences total, with `omomo_correct_v2`
being the curated-and-fixed OMOMO), we faced a choice:

- **Keep both**: CHOIS-OMOMO (4919) + InterAct (8478 – 4890 = 3588 new)
- **Drop CHOIS-OMOMO**: use InterAct exclusively, getting 8478 sequences via
  one unified preprocessing path

User chose the latter.

## Rationale

1. **`omomo_correct_v2` is the authoritative OMOMO.** The `correct_v2` suffix
   is the InterAct team's explicit statement that this version fixes issues
   present in the original OMOMO release.
2. **Unified preprocessing path.** One `preprocess_interact.py` covers all 4
   subsets with identical SMPL-X FK + HumanML3D encoder settings, eliminating
   two-path maintenance overhead.
3. **More data.** 8478 vs 4919 sequences. More importantly, chairs / imhd /
   neuraldome add 3588 sequences across **106 unique objects** (vs 15 for
   OMOMO alone). Much broader object-category coverage, directly relevant to
   PIANO's object-adaptive claim.
4. **CHOIS-OMOMO hadn't run pseudo-label extraction yet**, so no sunk cost in
   the CHOIS pipeline beyond the preprocessing itself (which we keep for
   reference).

## Results

### Format unification confirmed

All 4 InterAct subsets share:
- `human.npz`: `poses` (T, 156), `betas` ((10,) for chairs, (16,) for others),
  `trans` (T, 3), `gender`
- `object.npz`: `angles` (T, 3), `trans` (T, 3), `name`
- `text.txt`: natural-language description before `#` delimiter
- Object meshes under `objects/<name>/<name>.obj` + `sample_points.npy`

Only wrinkle: chairs ships 10 betas while other subsets ship 16.
`preprocess_interact._pad_betas` pads to 16 with zeros.

### Code changes

- **NEW** `src/piano/data/preprocess_interact.py` (unified preprocessor, ~470 lines)
- **NEW** `scripts/data/preprocess_interact.sh` (wrapper)
- **NEW** `scripts/data/extract_pseudo_labels_interact.sh` (iterates 4 subsets)
- **MODIFIED** `src/piano/data/pseudo_labels/run_all.py::_find_mesh` to handle
  InterAct's `<obj>/<obj>.obj` nested layout (in addition to the flat layout
  OMOMO used)
- **MODIFIED** defaults in `check_hoi_dataset.sh`, `visualize_motion.sh`,
  `inference_smoke_test.py` now point at `InterAct/piano/omomo_correct_v2`
- **Dropped from training plan** (but code retained): `preprocess_omomo.py`,
  `extract_pseudo_labels_omomo.sh`

### Dependency fix

Running the OMOMO pseudo-label extraction revealed that
`trimesh.proximity.closest_point` silently depends on `rtree` (and its C lib
`libspatialindex`). Without it, every sequence failed:

```
[warn] sub17_woodchair_056: No module named 'rtree'
Done. 0 labels written, 4919 skipped.
```

Fix: add `rtree` to conda environment (from conda-forge so libspatialindex
comes along automatically). Also recorded in `pyproject.toml`.

Server command:
```
conda install -c conda-forge rtree -y
```

This fix applies to BOTH OMOMO and InterAct pseudo-label extraction (same
underlying code path) — so even though we dropped CHOIS-OMOMO from the
training plan, the diagnostic from that run was useful.

## Observations

- The decision to drop CHOIS-OMOMO only cost us the preprocessing run (~4
  min on A6000). The rtree diagnostic it yielded is worth more than that.
- Having two coordinate frames in each `.npz` (HumanML3D canonical
  `motion_263` for VQ-VAE training + raw world-frame `joints_22` +
  `object_positions` for pseudo-labels) continues to be the right design —
  it's preserved in `preprocess_interact.py` identically.

## Action Items (→ PLAN.md)

- [x] Write and commit `preprocess_interact.py`
- [x] Add rtree to environment
- [ ] User: `conda install -c conda-forge rtree -y` on server
- [ ] User: run `preprocess_interact.sh --num-samples-limit 10` smoke test
- [ ] User: run full `preprocess_interact.sh` (8-10 min on cuda)
- [ ] User: run `extract_pseudo_labels_interact.sh` (1-3 hours CPU)
- [ ] Then: Stage A training with unified 4-subset data
