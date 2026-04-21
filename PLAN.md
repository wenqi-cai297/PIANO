# PIANO — Action Plan

Current priorities and next steps. Updated after each experiment analysis cycle.

**Last updated:** 2026-04-22 (v4 vis confirmed the 3 diagnostic sit-on-sofa clips still have `sitting=0`. Server diagnostic showed `neuraldome/bigsofa` is authored Z-up while other InterAct objects are Y-up — the fixed `normal.Y > 0.7` filter drops every sofa seat face. Below-gate now auto-detects the mesh up axis per object. 15/15 regression tests green (new Z-up test). v5 rerun queued behind this commit.)

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
- [x] **v1 rerun on server + stats** (2026-04-20, commits `34ccf3c`):
  8473 / 8475 sequences extracted, rich stats tool written + run.
- [x] **Stats exposed a deeper bug** (2026-04-20, commit `d641732`):
  v1 labels were unusable (81-99% zero-contact, 100% degenerate target).
  Three root causes recalibrated: (a) per-body-part distance thresholds
  with anatomy-calibrated values (hand 0.08 / foot 0.12 / pelvis 0.20
  meters; joint-to-skin offset, not joint penetration); (b) velocity
  gating off by default (was multiplying contact to zero); (c) target
  soft-assign kernel corrected from `-d/(2σ²)` to `-d²/(2σ²)`, sigma
  raised 0.01 → 0.05. See
  [pseudo_label_stats_v1_diagnosis](analyses/2026-04-20_pseudo_label_stats_v1_diagnosis.md).
- [x] **Full-dataset plain threshold sweep done** (2026-04-20, commit
  `4252130`). Curves confirmed hand 0.08 / pelvis 0.20 were anatomically
  right; foot threshold revised from 0.12 → 0.06 after seeing raw
  distance distributions. ContactConfig already matches (`d641732`).
- [x] **Stricter-prior validation via action window: abandoned**
  (2026-04-21). text.txt `#start#end` timestamps are placeholder 0.0 on
  every sampled file; real frame ranges live in per-dataset annotation
  CSVs but omomo_correct_v2 (58% of data) has no such CSV, and chairs /
  imhd / neuraldome would need non-trivial video_url → seq_id mapping.
  See [text_annotation_probe_dead_end](analyses/2026-04-21_text_annotation_probe_dead_end.md).
  Validation defers to quality_flags + visualization + Stage A
  held-out accuracy.
- [ ] **Re-run preprocess once to populate new SMPL-X fields** (`c7e9272`).
  - `piano-preprocess-interact` — adds `smplx_poses/trans/betas` to each
    `motions/<seq>.npz`. ~10 min on A6000. Required before Stage B if we
    want SDF penetration loss; harmless to defer until we need it.
- [x] **v2 pseudo-label extraction + visualization done** (2026-04-21 AM/PM)
  - Verdict: contact pass (chairs / imhd / omomo); neuraldome 49% over bar (hand threshold).
  - Target entropy: 4/4 below bar → sigma fix.
  - Surfaced 5th P0 bug (sitting FP for push/drag) via visualization.
- [x] **Commit 7 fixes + start v3 rerun** — `a8f5c2e` landed, v3 done (~5 h).
- [x] **v3 summary.json judged** (2026-04-21 PM)
  - chairs sitting 64%→46% ✓ (dual gate filtered the push/drag FPs).
  - all 4 subsets `manipulation reached > 30%` ✓.
  - all 4 subsets `target entropy mean > 1.2` ✓ (chairs 0.26→1.21).
  - HMM NaN exceptions 5→0; 0 seqs skipped (was 5 in v2).
  - neuraldome zero-contact still 49% → hand threshold issue confirmed, deferred.
- [x] **v3 visualization + 14 clip spot-check** (2026-04-21 PM)
  - Dirs under `runs/visualizations/2026-04-21_16*` / `17*`.
  - Exposed below-gate over-rejection on sofa-edge sitting (3 clips where text says "sits"
    got `sitting=0`). Gate rewritten from closest-point-direction to cylinder + upward
    normal. 14/14 regression tests (+1 new test for sofa-edge).
- [x] **Below-gate rewrite committed** (`480762c`) → v4 rerun launched on server.
- [x] **v4 pseudo-label extraction done + aggregate judged** (2026-04-21 PM)
  - Phase/target/contact byte-identical to v3 (only support path changed) ✓ as designed.
  - chairs sitting 46.1%→49.6% (+3.5 pp); sofa-edge sits partially recovered.
  - neuraldome sitting 1.48%→1.60% (+0.12 pp) — small lift, likely the 0.7 upward-normal
    threshold rejects curved sofa cushion surfaces. Will confirm with v4 vis.
  - omomo sitting 0.10%→0.06% (noise level — omomo has no sitting content).
- [x] **v4 visualization done** (2026-04-22 02:18-02:27). 3 diagnostic sit-on-sofa
  clips still at `sitting=0`. Server face-normal probe on `bigsofa`, `neuraldome/chair`,
  `chairs/141`, `chairs/116` confirmed: bigsofa is Z-up (+Z 48901 >> -Z 7411), others
  Y-up. Hard-coded Y-up filter was the bug.
- [x] **Below-gate auto-detects the mesh up axis** (this commit). `_detect_mesh_up_axis`
  picks the cardinal +axis with the most seat-like face area; cylinder test is
  expressed along that axis instead of +Y. New regression test
  `test_support_auto_detects_up_axis_for_z_up_mesh` guards it.
- [ ] **NEXT: Commit + start v5 rerun** (same command, ~7 h). Phase/target/contact
  stay identical to v3/v4; only support (specifically sitting for Z-up meshes) changes.
- [ ] **v5 summary.json → check bigsofa sitting recovers**
  - Expect neuraldome sitting to rise from 1.60% (v4) toward 3-5% — all bigsofa
    seqs with "sit" in their text should now record non-zero sitting frames.
  - chairs sitting should stay ≈ 49.6% (chairs meshes were already Y-up, so
    auto-detect is a no-op for them).
- [ ] **If v5 still underperforms for bigsofa** → double-check mesh authoring or
  relax `sitting_below_upward_normal_threshold` 0.7 → 0.5.
- [ ] **Decide on remaining P1/P2 scope** (only after v5).
  See §3.1 for deferred list (hand threshold, suitcase mesh, expanded joints, etc).
- [ ] Update `configs/training/predictor.yaml` — multi-root InterAct paths,
  `fps=20`, `support_weight=0.1` until v5 confirms support labels.

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

### 3.1 Pseudo-label P1/P2 items (gated on v2 rerun stats + videos)

From the Codex review, these are real but unvalidated concerns. Fix
only if the v2 rerun outputs fail the relevant sanity check. Don't
prophylactically rewrite code.

| Item | Status | Gate condition |
|---|---|---|
| Soft-assignment kernel `-d / (2σ²)` → `-d² / (2σ²)` | ✅ Fixed in `d641732` (v1 stats showed 100% degenerate target) | — |
| Soft-assignment sigma 0.05 → 0.12 | ✅ Fixed locally 2026-04-21 PM (pending commit). v2 stats: entropy_mean 0.26 / 2.77max on chairs (60% degenerate) — kernel still too sharp. σ=0.12 brings neighbour patch mass to ~0.1 (up from ~4e-6). | — |
| Velocity gating on world-frame speed | ✅ Disabled by default in `d641732` (was killing contact; re-enable via `use_velocity_gating=True` for ablations) | Re-enable only for the ablation experiment |
| **Phase `is_contact` hand-only → any-body-part** (new, found 2026-04-21 AM) | ✅ Fixed locally 2026-04-21 (pending commit). Validated on v2 vis: `Sub0012_Obj116_Seg0_105` 100% sitting stuck in 100% approach; `subject02_bigsofa_0` sitting 55% but phase approach 70%. | — |
| **Phase `obj_vel` translation-only → translation + angular** (new, found 2026-04-21 AM) | ✅ Fixed locally 2026-04-21 (pending commit). Validated on v2 vis: `bat_righthand_swing_4_0` (reported as "rotating the bat continuously") had 47/80 frames in approach+stable-contact — P0-1+P0-2 compound. | — |
| HMM state → phase-name remapping | ✅ Fixed locally 2026-04-21 (pending commit) via `GaussianHMM(params="")` — M-step frozen, state k stays bound to phase k. | — |
| HMM NaN-fatal sequences | ✅ Fixed locally 2026-04-21 PM (pending commit). v2 aborted 5/8475 seq with "startprob_ must sum to 1 (got nan)". try/except around fit/predict now falls back to heuristic labels rather than dropping the sequence. | — |
| Replace `median_filter` on categorical labels with majority filter | ✅ Fixed locally 2026-04-21 (pending commit). `_majority_filter` via `np.bincount.argmax`. | — |
| **Support `sitting` false positive for push/drag seq** (new, found 2026-04-21 PM via vis) | ✅ Fixed in `a8f5c2e` (velocity gate) + `480762c` (below gate v2) + pending commit (auto-detect up axis). Two conjunctive gates: (i) pelvis XZ-plane speed < 0.15 m/s (1 s moving average) — rejects moving-while-pushing; (ii) an upward-facing seat surface must sit within a cylinder below the pelvis (XZ radius 0.15 m, vertical gate 0.30 m). The up direction is auto-detected per mesh because InterAct authors objects with mixed conventions (`neuraldome/bigsofa` is Z-up, chairs are Y-up). 6 new tests (velocity: push rejected / stationary preserved; below: too-far-above rejected / above-seat accepted / sofa-edge accepted / Z-up mesh accepted). | — |
| Contact velocity uses world frame instead of object-relative | Deferred; partial fix by disabling gating. Full object-relative velocity is only relevant if phase needs it to distinguish manipulation vs. stable. | v3 phase distribution shows `manipulation` < 30% of reached sequences even after rotation-aware fix |
| **Hand threshold 0.08m too strict for irregular / large objects** (new, found 2026-04-21 PM via vis) | Deferred. Visualization caught 4 zero-contact seq that actually had contact: `bat_holdhead_hit` (holding bat by thick end — wrist farther from surface), `suitcase_lefthand_push` (handle possibly missing from mesh), `neuraldome/box_1565` (holding big box with arms outstretched), `neuraldome/pan_360` (holding pan, wrist far from pan surface). | v3 imhd zero-contact frac > 20% or specific seq types fail; then try hand threshold 0.10-0.12 or add elbow/palm tracked joints |
| **InterAct `suitcase` mesh may omit the handle** (data-layer issue, found 2026-04-21 PM via vis) | Deferred, data-layer. `suitcase_lefthand_push` seq has user pushing the handle but object point cloud may not include it. Need to `mesh.bounds` vs `object_pc.bounds` comparison to verify. | Any suitcase-dependent training signal is suspect; if we depend on suitcase seq for Stage A support/target, check mesh completeness first |
| Expand tracked joints (elbows/knees/hips/spine) | Deferred | v3 sitting is still < 10% for chairs even after raising `pelvis` threshold to 0.25; means joint proxies aren't enough |
| Rename `support` → `foot_support` + add `unknown` class | Deferred | v3 `sitting` is still near-zero for chairs (confirms support is permanently foot-centric), or generator learns bad support behaviour in Stage C |
| Closest-surface-point for target (instead of joint position) | Deferred | v3 videos show visibly-wrong patch ids at contact frames |
| Hierarchical support redesign (foot_ground / body_object / support_type) | Long-term | Only if v1 support supervision degrades generator output after Stage C |

---

## 4. Timeline

| Week | Milestone | Status |
|------|-----------|--------|
| 1-2 | Environment + OMOMO prep on server | **Done** (2026-04-19) |
| 2 | InterAct prep (unzip → inspect → preprocess) | **Done** (2026-04-19) — 8475 seq across 4 subsets |
| 3 | Pseudo-labels extracted + validated (v2 running 2026-04-21; v3 queued behind commit of 4 phase/support P0 fixes) | **In progress** |
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
