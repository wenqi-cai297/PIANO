# OMOMO Preprocessing: SMPL-X FK → HumanML3D 263-dim — 2026-04-19

## Context

Converted all CHOIS-format OMOMO sequences to PIANO's internal format
(HumanML3D 263-dim + 22 joints + object point clouds) so they can be
consumed by `HOIDataset`.

- Input: `/media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/processed_data`
- Output: `/media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/piano`
- Pipeline: `src/piano/data/preprocess_omomo.py`, wrapper `scripts/data/preprocess_omomo.sh`
- Runner: A6000 GPU, cuda FK, bf16 environment

## Results

```
Processed sequences:   4919 (4380 train + 539 test)
Skipped sequences:      963 (all vacuum + mop — two-part objects)
Text coverage:         4838 / 4919 = 98.4%
Object point clouds:     13 (1024 points each, FPS-sampled from .obj)
Runtime:                 ~1 minute total (57s train + 6s test)
Throughput:              ~92 sequences/sec (GPU FK bottleneck)
```

Per-sequence outputs in `piano/motions/*.npz`:
- `motion_263`: `(T', 263)` float32 — HumanML3D features (T' ≈ T × 2/3 after downsample)
- `joints_22`: `(T', 22, 3)` float32 — joint world positions
- `object_positions`: `(T', 3)` float32 — per-frame object center

Metadata at `piano/metadata.json`: 4919 entries with `seq_id`, `split`,
`object_id`, `gender`, `text`, `num_frames`.

## Observations

### One non-trivial bug found and fixed

**SMPL-X batch size mismatch.** First attempt failed with:

```
Sizes of tensors must match except in dimension 1. Expected size 132 but got size 1
```

Root cause: `smplx.create(batch_size=1)` creates default zero buffers for
`jaw_pose`, `leye_pose`, `reye_pose`, `left_hand_pose`, `right_hand_pose`,
`expression` at `batch_size=1`. When forward was called with batch=132, these
defaults didn't match.

Fix in `src/piano/data/smplx_fk.py`: explicitly pass batch-sized zero tensors
for all unused SMPL-X params. Now forward works with any batch size.

### Downsampling: 30 → 20 fps via linear interp

CHOIS mocap is 30fps, HumanML3D/MoMask convention is 20fps. We apply linear
interpolation per-feature along the time axis (`downsample_temporal` in
`preprocess_omomo.py`). A 132-frame 30fps clip becomes 88 frames at 20fps.

### Two-part objects filtered (documented)

Following CHOIS default. See `2026-04-19_omomo_data_inspection.md` for why.
963 sequences (~16%) lost. Controlled by `PreprocessConfig.skip_objects` —
not a PIANO design decision.

### A6000 + cuda FK is very fast

SMPL-X forward kinematics on GPU + chunking (512 frames/chunk) processes ~92
sequences per second. Total 5882 sequences in under a minute.

## Diagnosis

The pipeline is sound. The two important implementation details (SMPL-X zero
buffers + temporal interpolation) are now captured as standard patterns and
can be reused for other HOI datasets (BEHAVE, InterAct, GRAB) with minor
field remapping.

## Implications

- Stage A training data is ready.
- Pseudo-label extraction (next step) can read these `motion_263` /
  `joints_22` outputs directly — no re-preprocessing needed.
- For InterAct (when it arrives), we can reuse `smplx_fk.py` verbatim and
  write a new `preprocess_interact.py` with dataset-specific field mapping.

## Action Items (→ PLAN.md)

- [x] Preprocess OMOMO on server (done)
- [x] Record 963-sequence filter reason in PROGRESS.md (done)
- [ ] Next: sanity-check `HOIDataset` can load this output
- [ ] Next: extract pseudo-labels on the 4919 sequences
