# PIANO — Action Plan

Current priorities and next steps. Updated after each experiment analysis cycle.

**Last updated:** 2026-04-19 (switched to InterAct-only data track, dropped CHOIS-OMOMO; rtree dep fix queued for the server)

---

## 1. Immediate Next Steps

### 1.1 Server-side Setup (blocking everything)

- [x] Clone PIANO on GPU server (done on A6000)
- [x] Clone MoMask backbone into `src/piano/models/backbones/momask`
- [x] Create conda env + install package (`pip install -e ".[wandb,viz,dev]"`)
- [x] Install OpenAI CLIP (`pip install ftfy regex git+https://github.com/openai/CLIP.git`)
- [x] Configure accelerate (bf16, RTX A6000 supports it)
- [x] Download MoMask HumanML3D checkpoints (188MB)
- [x] **Verify MoMask weight loading:** `bash scripts/server/check_momask_weights.sh` — all three models load cleanly
- [ ] PyTorch3D: skipped (not currently needed — object_encoder is pure PyTorch)
- [ ] Download HumanML3D mean/std and eval model for metrics (still needed for FID/R-Precision evaluation)

### 1.2 Data Preparation

**Data track: InterAct only (CHOIS-OMOMO path retired)**

Decision (2026-04-19): drop `preprocess_omomo.py` output from training plan —
`omomo_correct_v2` inside InterAct is the same dataset curated further by the
InterAct team, and using only InterAct gives us a single unified preprocessing
path across all 4 subsets (chairs / imhd / neuraldome / omomo_correct_v2).
Total sequences: 8478 (vs 4919 from CHOIS-OMOMO alone).

- [x] Download `InterAct.zip` + unzip
- [x] Inspect format (`check_interact_format.sh`); schema is uniform across subsets
- [x] Write `preprocess_interact.py` (reuses `HumanML3DEncoder` + `run_smplx_fk`)
- [x] Format-verification: 4 subsets detected, sample human/object npz keys match expectations
- [x] Fix rtree dependency (trimesh.proximity.closest_point needs spatial index)
  - Added to `environment.yml` + `pyproject.toml`; server needs
    `conda install -c conda-forge rtree -y`
- [ ] **NEXT: run preprocess_interact smoke test** (10 seq / subset)
  - `bash scripts/data/preprocess_interact.sh --num-samples-limit 10 --device cuda`
- [ ] Full InterAct preprocessing
  - `bash scripts/data/preprocess_interact.sh --device cuda`
  - Expected: ~8-10 minutes for 8478 sequences
- [ ] Extract pseudo-labels on all 4 subsets
  - `bash scripts/data/extract_pseudo_labels_interact.sh`
  - Expected: 1-3 hours CPU (trimesh + HMM across 8478 sequences)
- [ ] Visualize 5-10 samples per subset after pseudo-label extraction

**CHOIS-OMOMO path is retired but preserved:**
- `preprocess_omomo.py` + `extract_pseudo_labels_omomo.sh` kept in repo
  (reference implementation for CHOIS-style joblib datasets). No longer the
  primary training data path.

### 1.3 Code Gaps to Fill On-Server

These are trivial wiring that requires the actual environment:

- [ ] In `train_predictor.py` / `train_joint.py`: wire up `clip.load("ViT-B/32")` for text encoding
- [ ] In `train_joint.py`: load Stage A and Stage B checkpoints from config
- [ ] In `inference/generate.py`: implement checkpoint loading in `main()`

---

## 2. Training Roadmap

### Stage A — Interaction Predictor (1-2 days on A100)

- [ ] Run smoke test with 100 samples, verify loss decreases
- [ ] Full training on InterAct + CORE4D combined
- [ ] Evaluate predictor accuracy on held-out set (contact F1, phase accuracy, support accuracy)
- [ ] **Decision point:** if accuracy is low (<70%), revisit pseudo-label quality before continuing

### Stage B — Motion Generator Finetune (2-3 days on A100)

- [ ] Smoke test: load MoMask weights, train for 100 steps, verify CE loss behaves
- [ ] Full finetune with GT pseudo-labels as interaction condition
- [ ] Evaluate: FID, R-Precision on HumanML3D test, contact P/R/F1 on HOI test

### Stage C — Joint Finetune (1-2 days on A100)

- [ ] Load Stage A + B checkpoints
- [ ] Joint finetune with consistency loss
- [ ] Full evaluation suite: motion + physics + controllability metrics

### Ablations

- [ ] w/o interaction latent (end-to-end baseline)
- [ ] w/o phase variable
- [ ] w/o support variable
- [ ] w/o contact target
- [ ] w/o object attribute embedding
- [ ] w/o physical priors
- [ ] w/o consistency loss
- [ ] w/o Block AttnRes (standard residuals)

---

## 3. Decisions Pending

Open questions to resolve as we get experimental feedback:

1. **HOI data scale:** Is ~30h enough for the predictor to generalize across object attributes?
   - If not: data augmentation via object mesh scaling, synthetic text prompt variation
2. **Interaction phase granularity:** 5 phases or finer?
   - If pseudo-labels for `manipulation` vs `stable-contact` are too noisy, merge them
3. **Support state definition:** 4 states or more?
   - Current: both_feet, single_foot, sitting, hand_support
   - May need to add: `airborne` for jumping actions
4. **Block AttnRes block_size:** current 2 (5 blocks for 10 layers). Try 1 (Full AttnRes) if data permits?
5. **LaMP vs CLIP text encoder:** start with CLIP (MoMask compatibility), consider LaMP as improvement
6. **Residual Transformer:** currently frozen. Consider finetuning if contact details are lost?

---

## 4. Timeline

| Week | Milestone | Status |
|------|-----------|--------|
| 1-2 | Environment + OMOMO prep on server | **Done** (2026-04-19) |
| 2 | InterAct prep (unzip → inspect → preprocess) | **In progress** — downloaded, not yet unzipped |
| 3 | Pseudo-labels extracted (OMOMO first, InterAct follows) | **Next up** |
| 4-5 | Interaction Predictor trained and evaluated | Not started |
| 6-7 | Motion Generator finetuned with interaction conditioning | Not started |
| 8 | Joint finetune complete | Not started |
| 9-10 | Full evaluation + ablations + visualizations | Not started |

---

## 5. Risk Watch

| Risk | Trigger | Mitigation |
|------|---------|------------|
| Pseudo-label noise | Contact/phase accuracy < 70% on held-out | Soft labels + more aggressive temporal smoothing + region-level supervision |
| Generator ignores z_int | ASS score near baseline MoMask | Increase `interaction_scale` in CFG, strengthen consistency loss weight, dropout training |
| SMPL-X → SMPL info loss | Hand-object tasks fail | Defer hand-level tasks to v2; focus on full-body pick/place/sit for v1 |
| InterAct data access issues | Google Form delays | Start with CORE4D (open) + OMOMO as sanity check |
| MoMask weight loading fails | `load_state_dict` errors | Inspect exact key names via `_remap_key`; adjust if MoMask updates format |
| Attribute sensitivity not significant | ASS close to zero across attribute pairs | Augment training data by scaling object meshes; add per-attribute text variants |
| Reviewer: "incremental over Move as You Say" | — | Frame paper around object-adaptive problem, lead with ASS/ASC metrics |

---

## 6. Analysis → Plan Loop

This document is updated after each experiment analysis:

1. Download results to local → give files to Claude
2. Claude writes analysis to `analyses/YYYY-MM-DD_<topic>.md`
3. Claude adds entry to `ANALYSIS.md` index
4. Claude updates this PLAN.md with next steps based on findings
5. Commit + push all three files together
