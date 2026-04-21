# PIANO — Implementation Progress

Tracks what has been built, tested, and merged into the repository.
Updated after each significant code change.

**Last updated:** 2026-04-21 PM (v2 extraction complete + visualization reviewed on 14 representative seq. 4 original P0 fixes validated; visualization exposed a 5th P0-class bug — pelvis threshold → `support=sitting` false positive on push/drag sequences; fixed with two conjunctive gates (pelvis XZ velocity + object geometrically below pelvis). Total **7 local fixes** now pending commit (4 P0 + support sitting dual gate + target sigma 0.05→0.12 + HMM NaN fallback), 13/13 regression tests green. Thresholds from `127f675`; stricter-prior via text.txt abandoned)

---

## 0. Active Long Runs

Status of multi-hour jobs. Update when starting, finishing, or hitting a
checkpoint, so a cold-started assistant (post-`/clear`) can tell what's
currently executing versus what just landed. When a row goes stale
(>1 week since completion, no follow-up), compact it out.

| run | command | started | status | output / notes |
|---|---|---|---|---|
| v2 preprocess | `piano-preprocess-interact` | 2026-04-21 | ✓ done (7 min, 8475 seq) | each `<piano>/<subset>/motions/<seq>.npz` now has `smplx_poses` / `smplx_trans` / `smplx_betas` alongside existing fields |
| v2 pseudo-label extraction | `bash scripts/data/rerun_pseudo_labels_interact.sh` | 2026-04-21 | ✓ done (~5 h on server); judged | contact/target/phase/support stats + quality_flags in `<piano>/<subset>/pseudo_labels/summary.json`. Verdict: contact pass (except neuraldome 49% zero-contact — hand-threshold issue, deferred); target 4/4 below entropy pass bar → triggered sigma 0.05→0.12 local fix; phase/support fields untrustworthy until v3 (P0 fixes). See [analyses/pseudo_label_phase_support_fixes](analyses/2026-04-21_pseudo_label_phase_support_fixes.md) §a-c. |
| v2 visualization | `bash scripts/server/visualize_finished_subsets.sh` | 2026-04-21 PM | ✓ done, 14 seq spot-checked | Videos at `runs/visualizations/2026-04-21_*/`. Validated 4 P0 fixes on chairs / imhd / neuraldome seq; exposed 5th bug (pelvis false positive for push/drag) — local fix applied. |
| v3 pseudo-label extraction | (queued) | pending commit of 7 local fixes | ⏸ queued | Same command as v2. Phase/support fields will be trustworthy in v3 output. |

---

## 1. Module Status Overview

| Module | Files | Status | Verification |
|--------|-------|--------|--------------|
| **Project scaffolding** | pyproject.toml, environment.yml, configs/ | ✓ Done | `pip install -e .` succeeds |
| **Utils** | io_utils, geometry, smpl_utils | ✓ Done | Unit tests passed |
| **Data processing** | humanml3d_repr, preprocess_smplx, preprocess_interact, dataset | ✓ Done (v2, 2026-04-21) | SMPL-X → 22 joints → 263-dim conversion verified; preprocess now also preserves full `smplx_poses` / `smplx_trans` / `smplx_betas` per sequence for downstream mesh-based losses. v2 re-run completed 2026-04-21 (8475 seq / 7 min) — new fields populated. |
| **Pseudo-label extraction** | extract_contact/target/phase/support, refine_hmm, run_all | ⚠ v2 done + visualised; **7 local fixes** pending commit → v3 | Post-sweep thresholds: hand 0.08 / foot 0.06 / pelvis 0.20 m (`127f675`). v1 stats fixed 3 bugs in `d641732`. 2026-04-21 AM code review: 4 P0 design bugs (any-body-part contact, rotation-aware phase, frozen HMM, majority support filter). 2026-04-21 PM visualization validated those 4 on 14 seq + exposed **5th P0 bug**: pelvis threshold caused `support=sitting` false positive on push/drag seq (neuraldome bigsofa_330 / chair_0 standing & pushing were labelled sitting 96% / 63%). Fixed with two conjunctive gates on `sitting`: (i) pelvis XZ velocity < 0.15 m/s (1 s moving average) — rejects moving-while-pushing; (ii) object geometrically below pelvis — pelvis→closest-point direction has Y-component < -0.3 — rejects standing-beside-object-stationary. Also added target sigma 0.05→0.12 (v2 entropy 4/4 below pass bar) and HMM NaN fallback (v2 had 5/8475 exceptions). 13/13 regression tests (`tests/test_pseudo_labels.py`). fps propagation + deterministic patch atlas unchanged. See §0 and [analyses/pseudo_label_phase_support_fixes](analyses/2026-04-21_pseudo_label_phase_support_fixes.md). |
| **Object Encoder** | object_encoder.py (PointNet++) | ✓ Done | Forward pass OK, 0.3M params, feature_dim=384 |
| **Interaction Predictor** | interaction_predictor.py | ✓ Done | 10 layers, d=384, Block AttnRes (5 blocks), 31.8M params |
| **Interaction Cross-Attention** | interaction_cross_attn.py | ✓ Done | Zero-init verified |
| **Interaction Extractor** | interaction_extractor.py | ✓ Done | Forward pass OK, 2.5M params |
| **Motion Generator** | motion_generator.py (thin wrapper) + masking.py | ✓ Done | Wraps MoMask's `MaskTransformer`; patches `seqTransEncoder`; 100% weight-compat |
| **MoMask Adapter** | backbones/momask_adapter.py | ✓ Done | Imports MoMask's original classes; `load_momask_vqvae/mask_transformer/residual_transformer` |
| **MoMask Backbone** | backbones/momask/ | ✓ Done | git cloned, gitignored |
| **Training: Losses** | losses.py | ✓ Done | PredictorLoss, GeneratorLoss, ConsistencyLoss |
| **Training: Priors** | priors.py | ✓ Done | Reachability, contact persistence, support smoothness, phase monotonicity |
| **Training: Shared** | trainer.py | ✓ Done | Accelerate loop, checkpoints, wandb, cosine LR + warmup |
| **Training: Stage A** | train_predictor.py | ✓ Done (skeleton) | Predictor + priors; CLIP loading is TODO |
| **Training: Stage B** | train_generator.py | ✓ Done (skeleton) | Frozen VQ-VAE, dual-LR via backbone_parameters/interaction_parameters |
| **Training: Stage C** | train_joint.py | ✓ Done (skeleton) | Predicted z_int + consistency loss |
| **Evaluation: Motion** | motion_metrics.py | ✓ Done | FID, R-Precision, MM-Dist, Diversity, MultiModality |
| **Evaluation: Physics** | physics_metrics.py | ✓ Done | Penetration, contact P/R/F1, foot sliding, support consistency |
| **Evaluation: Controllability** | controllability.py | ✓ Done | ASS, ASC, latent sensitivity |
| **Inference** | generate.py (PIANOPipeline) | ✓ Done (skeleton) | End-to-end text+object → motion |
| **Inference: Viz** | visualize.py | ✓ Done | motion_263 → joints, skeleton frame rendering |
| **Tests** | tests/test_pseudo_labels.py | ✓ 13/13 passing (2026-04-21 PM, local `piano` conda env) | Regression tests locking all 7 local fixes: any-body-part contact for phase (3 tests), rotation-aware obj motion, frozen HMM state ids, majority (not median) support filter, sitting dual gate — velocity (push rejected + stationary preserved) AND object-below (beside-object rejected + above-object preserved), target sigma 0.12 gives soft distribution, HMM NaN fallback |

**Total:** 38 Python files, ~5400 lines of code (excluding MoMask backbone).

---

## 2. What's Runtime-Ready on Server

When cloned on a GPU server with the environment set up, the following can run:

- `pip install -e ".[wandb,viz,dev]"` → package installs cleanly
- `from piano.models.* import *` → all models importable
- Forward passes on all models verified with synthetic inputs
- `piano-pseudo-labels --data-dir ... --output-dir ...` → CLI entrypoint registered
- `piano-eval`, `piano-generate`, `piano-train` → CLI entrypoints registered
- `piano-check-momask` / `bash scripts/server/check_momask_weights.sh` → **verified on A6000 server:** all three MoMask pretrained checkpoints (VQ-VAE 19.4M, MaskTransformer 163.3M, ResidualTransformer 164.6M) load cleanly without warnings

### Data preparation

**OMOMO (via CHOIS processed_data)**
- Source: `/media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/processed_data` (30fps SMPL-X mocap, 17 subjects × 15 objects)
- Preprocessed to PIANO format at `/media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/piano`
- Pipeline (v2, post encoder-switch): joblib load → SMPL-X FK → 22 body joints → downsample 30→20fps → **MoMask `process_file` encode** → HumanML3D-compatible 263-dim + raw world-frame joints + object positions
- **Result: 4919 sequences (4380 train + 539 test), 4838 with text (98.4% coverage), 13 object point clouds**
- **Skipped: 963 sequences involving `vacuum` and `mop`** — two-part articulated objects; CHOIS's default behavior inherited via `PreprocessConfig.skip_objects`. Articulated-object handling is out of scope for PIANO v1.
- Runtime: ~4 minutes on single A6000 (cuda FK + CPU uniform-skeleton IK)
- **Two coordinate frames preserved side-by-side** (intentional):
  - `motion_263`: HumanML3D canonical + uniform skeleton (for MoMask VQ-VAE)
  - `joints_22` + `object_positions`: raw world frame (for pseudo-label geometry)
- See analyses: [omomo_data_inspection](analyses/2026-04-19_omomo_data_inspection.md),
  [omomo_preprocessing](analyses/2026-04-19_omomo_preprocessing.md),
  [hoi_dataset_verification](analyses/2026-04-19_hoi_dataset_verification.md),
  [momask_weight_loading](analyses/2026-04-19_momask_weight_loading.md),
  [inference_smoke_test](analyses/2026-04-19_inference_smoke_test.md),
  [humanml3d_encoder_switch](analyses/2026-04-19_humanml3d_encoder_switch.md)

**InterAct (CVPR 2025) — primary data track**
- Downloaded + unzipped at `/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/InterAct/`
- Format inspected via `check_interact_format.sh` — 4 subsets share a uniform schema
  (human.npz with poses/betas/trans/gender, object.npz with angles/trans/name, text.txt)

| Subset | Sequences | Unique objects |
|--------|----------:|---------------:|
| chairs | 1502 | 60 |
| imhd | 595 | 10 |
| neuraldome | 1491 | 21 |
| omomo_correct_v2 | 4890 | 15 |
| **Total** | **8478** | **106** |

- `preprocess_interact.py` written: shared `HumanML3DEncoder` + SMPL-X FK
  pipeline with pose-splitting [0:3] root / [3:66] body (hands ignored for v1)
  and betas padded to 16 (chairs ships 10)
- Each subset writes to its own PIANO root (`/media/.../InterAct/piano/<subset>/`)
  so `HOIDataset` can combine multiple roots at training time
- `extract_pseudo_labels_interact.sh` iterates all 4 subsets
- `_find_mesh` extended to handle InterAct's nested `objects/<name>/<name>.obj` layout
- **Full preprocessing run 2026-04-19 08:25:12**: 8475/8478 sequences processed
  in 9.4 min on A6000, 100% text coverage, 106 unique objects. Only 3 seqs
  in imhd skipped (0.04% failure). See
  [interact_preprocessing_complete](analyses/2026-04-19_interact_preprocessing_complete.md).

**CHOIS-OMOMO path: retired.** 4919-sequence preprocessing was successful but
`omomo_correct_v2` inside InterAct supersedes it. `preprocess_omomo.py` code
retained for reference / future CHOIS-style datasets.

**Dependency note**: `rtree` (required by `trimesh.proximity.closest_point`)
must be installed from `conda-forge` so `libspatialindex` comes along.
Added to `environment.yml`.

### End-to-end inference baseline

- Ran `inference_smoke_test.sh` on 4 OMOMO samples (2026-04-19 063940)
- All shapes verified; output finite; token_ids within valid VQ range
- Zero-init interaction cross-attn confirmed to preserve MoMask behavior
- Fixed: device mismatch where new wrapper layers stayed on CPU
- **Encoder round-trip validated (2026-04-19)**: after switching to MoMask's
  `process_file`, generated videos show proper humanoid structure (pelvis
  at correct height, skeleton coherent); real samples via `--use-recovery`
  path also recover correctly. This confirms `motion_263` is byte-compatible
  with MoMask's VQ-VAE input distribution.

---

## 3. Known TODOs Inside the Code

These are marked as `TODO` in the source — trivial on-server wiring, not design gaps:

- **CLIP loading** in training scripts: needs `clip.load("ViT-B/32")` call with device setting
  - Files: `train_predictor.py`, `train_joint.py`
- **Stage A/B checkpoint loading** into Stage C joint finetune
  - File: `train_joint.py`
- **Inference checkpoint loading** from disk
  - File: `inference/generate.py` — `main()` raises NotImplementedError until checkpoints exist

All other components are functionally complete.

---

## 4. Commit History (major milestones)

| Commit | Date | What |
|--------|------|------|
| `1be625e` | 2026-04-13 | Initial commit: project spec for PIANO |
| `c4aeb57` | 2026-04-13 | Add core codebase: data pipeline, models, MoMask-compatible generator |
| `f130bef` | 2026-04-13 | Fix configs to match MoMask pretrained checkpoint dimensions |
| `bd662fc` | 2026-04-13 | Complete training, evaluation, and inference modules |
| `d31a08d` | 2026-04-14 | Refactor motion_generator to use MoMask source directly |
| `c8e5e06` | 2026-04-14 | Fix stale references to removed classes |
| `c02a388` | 2026-04-14 | Scale Interaction Predictor with Block Attention Residuals |
| `644929e` | 2026-04-19 | Add OMOMO preprocessing: SMPL-X FK + downsample + 263-dim (v1) |
| `1e8749a` | 2026-04-19 | HOIDataset sanity check + verified 4919 sequences load |
| `c947228` | 2026-04-19 | End-to-end inference smoke test with zero-init cross-attn |
| `9eb1c68` | 2026-04-19 | Fix device mismatch in InteractionMaskTransformer |
| `c115e30` | 2026-04-19 | Standardize: every script writes runs/<cat>/<ts>/summary.json |
| `031333e` | 2026-04-19 | Switch to MoMask official HumanML3D encoder (process_file) |
| `3250eb0` | 2026-04-19 | Add `--use-recovery` flag for encode→decode round-trip validation |
| `40c703b` | 2026-04-19 | Fix pseudo-label geometry: inverse-transform joints to object-local frame |
| `9d11f1a` | 2026-04-20 | Propagate fps + deterministic per-object patch atlas in pseudo-labels |
| `34ccf3c` | 2026-04-20 | Rich pseudo-label stats: contact/phase/support/target/geometry + quality flags |
| `d641732` | 2026-04-20 | Recalibrate contact + target for joint-to-skin offset and true soft assign |
| `4252130` | 2026-04-20 | Add piano-threshold-sweep: cache distances once, re-score per threshold |
| `c7e9272` | 2026-04-20 | Preserve full SMPL-X params in preprocess output (poses + trans + betas) |
| `dde931c` | 2026-04-20 | Add piano-action-segment-sweep (skeleton kept; stricter-prior path later abandoned) |
| `fd88445` | 2026-04-21 | Add piano-probe-text-annotations (format discovery probe) |
| `3811873` | 2026-04-21 | Docs: record stricter-prior dead end + finalise thresholds for v2 rerun |
| `127f675` | 2026-04-21 | Apply the foot-threshold fix (0.12 → 0.06) |
| `1b7cee8` | 2026-04-21 | Don't auto-kill piano-labels tmux session by default |

---

## 5. Environment Details

- **Git repo:** `git@github.com:wenqi-cai297/PIANO.git` (private, SSH)
- **Branch:** master
- **Package name:** `piano` (installed via `pip install -e .`)
- **CLI entrypoints:** `piano-train`, `piano-eval`, `piano-generate`, `piano-pseudo-labels`
- **Training framework:** HuggingFace Accelerate
- **Logging:** wandb
- **Config:** OmegaConf yaml files under `configs/`
- **Local dev env** (Windows, Miniforge3): conda env `piano` built 2026-04-21 — python 3.10, rtree (conda-forge), numpy/scipy/scikit-learn/hmmlearn/pytest (pip), `piano` editable via `pip install -e . --no-deps`. Minimal for pseudo-label pytest runs; heavier deps (torch / trimesh / smplx / open3d / CLIP) are added on demand when other modules need smoke-testing locally.
