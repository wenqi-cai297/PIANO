# PIANO — Action Plan

Current priorities and next steps. Updated after each experiment analysis cycle.

**Last updated:** 2026-04-20 (Codex review integrated; P0 fixes landed in `9d11f1a`; rerun + stats + qualitative inspection is the next gate before any P1/P2 work or Stage A)

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
- [x] **Full InterAct preprocessing done** (2026-04-19 08:25:12):
  8475 / 8478 sequences in 9.4 min on A6000, 100% text, 106 unique objects.
  See [analyses/interact_preprocessing_complete](analyses/2026-04-19_interact_preprocessing_complete.md).
- [x] **Pseudo-label pipeline P0 fixes** (2026-04-20, commit `9d11f1a`):
  fps now auto-resolved from preprocess summary (data is 20 fps, configs
  used to default to 30 → velocity thresholds were inflated 1.5×);
  per-object patch atlas is now deterministic (`seed=md5(obj_id)`) and
  disk-cached at `<output>/patch_atlas/<obj_id>.npy`.
  See [codex_review_p0_fixes](analyses/2026-04-20_codex_review_p0_fixes.md)
  and [`SUGGESTION.md`](SUGGESTION.md).
- [ ] **NEXT: rerun pseudo-label extraction on server**
  - `bash scripts/data/rerun_pseudo_labels_interact.sh` (kills old tmux,
    backs up previous `pseudo_labels/` dirs, re-extracts all 4 subsets)
  - Expected: ~1.5h CPU
- [ ] **Write stats aggregator** (while rerun is running):
  per-subset contact-rate per body part, phase/support histograms,
  patch-id entropy, short-contact fraction, HMM non-convergence count.
  Output to `runs/pseudo_labels_stats/<ts>/summary.json` + markdown.
- [ ] Visualize 5-10 samples per subset after rerun completes
  - Tool ready: `bash scripts/server/visualize_pseudo_labels.sh`
  - Overlays contact/phase/support on skeleton MP4
- [ ] **Decide on P1/P2 scope** (only after stats + videos in hand).
  See §3 for the deferred-until-validation list.
- [ ] Update `configs/training/predictor.yaml` with multi-root InterAct paths

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
   - May need to add: `airborne` for jumping actions, or `unknown` per Codex review
4. **Block AttnRes block_size:** current 2 (5 blocks for 10 layers). Try 1 (Full AttnRes) if data permits?
5. **LaMP vs CLIP text encoder:** start with CLIP (MoMask compatibility), consider LaMP as improvement
6. **Residual Transformer:** currently frozen. Consider finetuning if contact details are lost?

### 3.1 Pseudo-label P1/P2 items (gated on rerun stats + videos)

From the Codex review, these are real but unvalidated concerns. Fix
only if the rerun outputs fail the relevant sanity check. Don't
prophylactically rewrite code.

| Item | Gate condition (what would trigger the fix) |
|---|---|
| Contact velocity uses world frame instead of object-relative | Manipulation sequences show fragmented contact in videos, or contact-rate histograms for moving-object subsets (neuraldome, omomo_correct_v2) are materially lower than for static-object subsets (chairs) |
| Rename `support` → `foot_support` + add `unknown` class | Sitting/lying sequences in `chairs` show the `sitting` label firing inconsistently, or `both_feet` dominates frames where no body part is near the object |
| Replace `median_filter` on categorical labels with majority filter | Support/phase histograms show odd intermediate-class spikes (e.g. `single_foot` inflated at transitions) |
| Soft-assignment kernel `-d / (2σ²)` → `-d² / (2σ²)` | Patch-id entropy is near zero across contact frames (effectively hard nearest-patch) |
| Closest-surface-point for target (instead of joint position) | Target patch is visibly wrong at contact frames in videos (body joint is close but patch id is a far patch) |
| Expand tracked joints (elbows/knees/hips/spine) | Visible contacts in chair/sofa sequences are missed by the 5-joint extractor |
| HMM state → phase-name remapping | Phase histograms across subsets show a semantically incoherent state id dominating |
| Hierarchical support redesign (foot_ground / body_object / support_type) | Long-term; only if v1 support supervision degrades generator output after Stage C |

---

## 4. Timeline

| Week | Milestone | Status |
|------|-----------|--------|
| 1-2 | Environment + OMOMO prep on server | **Done** (2026-04-19) |
| 2 | InterAct prep (unzip → inspect → preprocess) | **Done** (2026-04-19) — 8475 seq across 4 subsets |
| 3 | Pseudo-labels extracted + validated (rerun with P0 fixes) | **In progress** (2026-04-20, rerun queued) |
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
