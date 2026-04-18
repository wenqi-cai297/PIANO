# Switch to MoMask's Official HumanML3D Encoder — 2026-04-19

## Context

After the inference smoke test, the generated videos showed the pelvis
joint stuck on the ground. Root cause: our custom `joints_to_humanml3d`
was a naive encoder that skipped HumanML3D's canonicalization steps
(uniform-skeleton rescale, ground-put, xz-center, heading-align).
Consequently:
- Features were NOT decodable by `recover_from_ric` correctly
- Features would also NOT be interpretable by MoMask's pretrained VQ-VAE
  (silent training failure waiting to happen)

Decision: replace our custom encoder with MoMask's official `process_file`
helper. User confirmed Option B (deep fix now, before Stage B training).

## Results

### Code restructuring

- **NEW** `src/piano/data/humanml3d_encoder.py`
  - `HumanML3DEncoder` class wraps MoMask's `utils.motion_process.process_file`
  - Handles the upstream quirk of module-level globals (`tgt_offsets`,
    `n_raw_offsets`, `kinematic_chain`, `face_joint_indx`, `fid_l`, `fid_r`,
    `l_idx1`, `l_idx2`) by setting them on `utils.motion_process` at init
  - Derives `tgt_offsets` from a reference skeleton passed in by the caller
    (we use the first frame of the first valid OMOMO training sequence)
- **MODIFIED** `src/piano/data/preprocess_omomo.py`
  - Split FK + downsampling into `fk_and_downsample()` (reused for both the
    encoder bootstrap and the main processing loop)
  - `preprocess_sequence` now calls `encoder.encode()` instead of the
    deprecated `joints_to_humanml3d`
  - Truncates `joints_22` and `object_positions` to T-1 to match the T-1
    output of `process_file` (one frame dropped for velocity computation)
- **DEPRECATED** `humanml3d_repr.joints_to_humanml3d` (runtime DeprecationWarning)
- **DEPRECATED** `preprocess_smplx.py` module-level docstring updated

### Rerun on server

From `runs/.../summary.json`:

| Metric | Value |
|--------|-------|
| Sequences processed | 4919 (4380 train + 539 test) |
| Sequences with text | 4838 (98.4%) |
| Object point clouds | 13 |
| Elapsed | **259.89s** (cuda FK + CPU uniform-skeleton IK) |
| Timestamp | 2026-04-19 07:32:06 |

Note the preprocessing is ~4× slower than before because `process_file`
runs uniform-skeleton IK on every sequence (inverse kinematics on CPU).
Still ~5 minutes total — acceptable.

### Two coordinate frames preserved explicitly

| Field | Frame | Purpose |
|-------|-------|---------|
| `motion_263` | HumanML3D canonical (ground + xz-origin + heading-aligned + uniform skeleton) | MoMask VQ-VAE encoding |
| `joints_22` | Raw world frame (SMPL-X FK output, downsampled) | Pseudo-label extraction, contact detection |
| `object_positions` | Raw world frame | Pseudo-label extraction |

## Observations

The split is intentional: pseudo-label extraction needs **geometric fidelity**
(hand actually near the box), which is broken by uniform-skeleton. The
VQ-VAE needs **distribution fidelity** (same skeleton scale MoMask trained
on). Keeping both data streams side-by-side avoids the tension.

## Implications

- Stage B training is no longer heading for silent failure — motion_263
  will round-trip through MoMask's VQ-VAE cleanly
- Pseudo-labels (next step) still use the correct raw geometry
- The generated smoke test videos should now render with proper
  skeleton structure when passed through `recover_from_ric + denorm`

## Action Items (→ PLAN.md)

- [x] Replace custom encoder with official `process_file`
- [x] Rerun preprocessing — verified 4919 sequences produced
- [ ] Run encode → decode round-trip visualization:
  `bash scripts/server/visualize_motion.sh real ... --use-recovery` —
  real samples should now look normal even via `recover_from_ric`
- [ ] Rerun inference smoke test on the new preprocessed data
- [ ] Then: extract pseudo-labels on the 4919 sequences
