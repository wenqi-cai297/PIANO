# PIANO — Action Plan

Current priorities and next steps. Updated after each experiment analysis cycle.

**Last updated:** 2026-04-19

---

## 1. Immediate Next Steps

### 1.1 Server-side Setup (blocking everything)

- [ ] Clone PIANO on GPU server: `git clone git@github.com:wenqi-cai297/PIANO.git`
- [ ] Clone MoMask backbone: `git clone https://github.com/EricGuo5513/momask-codes.git src/piano/models/backbones/momask`
- [ ] Create conda env: `conda env create -f environment.yml`
- [ ] Install package: `pip install -e ".[wandb,viz,dev]"`
- [ ] Install PyTorch3D
- [ ] Configure accelerate: `accelerate config`
- [ ] Download MoMask HumanML3D checkpoints (~188MB zip from Google Drive)
- [ ] Download HumanML3D mean/std and eval model for metrics

### 1.2 Data Preparation

- [ ] Download CORE4D from HuggingFace (open, no form)
- [ ] Apply for InterAct subsets that require Google Form access
- [ ] Run SMPL-X → SMPL 22-joint conversion on all data
  - `python -m piano.data.preprocess_smplx --input-dir ... --output-dir ...`
- [ ] Extract pseudo-labels on the full preprocessed dataset
  - `piano-pseudo-labels --data-dir data/processed --output-dir runs/pseudo_labels`
- [ ] Visualize 10-20 random samples to verify pseudo-label quality before training

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
| 1-2 | Environment + data prep on server | Not started |
| 3 | Pseudo-labels extracted and verified | Not started |
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
