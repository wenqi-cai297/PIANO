# PIANO — Implementation Progress

Tracks what has been built, tested, and merged into the repository.
Updated after each significant code change.

**Last updated:** 2026-04-19

---

## 1. Module Status Overview

| Module | Files | Status | Verification |
|--------|-------|--------|--------------|
| **Project scaffolding** | pyproject.toml, environment.yml, configs/ | ✓ Done | `pip install -e .` succeeds |
| **Utils** | io_utils, geometry, smpl_utils | ✓ Done | Unit tests passed |
| **Data processing** | humanml3d_repr, preprocess_smplx, dataset | ✓ Done | SMPL-X → 22 joints → 263-dim conversion verified |
| **Pseudo-label extraction** | extract_contact/target/phase/support, refine_hmm, run_all | ✓ Done | Phase + support unit tests passed |
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

**Total:** 38 Python files, ~5400 lines of code (excluding MoMask backbone).

---

## 2. What's Runtime-Ready on Server

When cloned on a GPU server with the environment set up, the following can run:

- `pip install -e ".[wandb,viz,dev]"` → package installs cleanly
- `from piano.models.* import *` → all models importable
- Forward passes on all models verified with synthetic inputs
- `piano-pseudo-labels --data-dir ... --output-dir ...` → CLI entrypoint registered
- `piano-eval`, `piano-generate`, `piano-train` → CLI entrypoints registered

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

---

## 5. Environment Details

- **Git repo:** `git@github.com:wenqi-cai297/PIANO.git` (private, SSH)
- **Branch:** master
- **Package name:** `piano` (installed via `pip install -e .`)
- **CLI entrypoints:** `piano-train`, `piano-eval`, `piano-generate`, `piano-pseudo-labels`
- **Training framework:** HuggingFace Accelerate
- **Logging:** wandb
- **Config:** OmegaConf yaml files under `configs/`
