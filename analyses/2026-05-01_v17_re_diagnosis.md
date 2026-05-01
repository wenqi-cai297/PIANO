# 2026-05-01 — v17 inference series re-diagnosis (source-level, local-first)

Phase 2 `/restart` re-diagnosis. Pressure-tests the prior diagnosis pinned at
`analyses/2026-05-01_v17g_gamma_int_boost_result.md`:

> v17 inference-side TTT path is saturated. γ_int is *undertrained*, not
> *under-applied*. Next branch is P2 (re-init γ_int + Stage B finetune,
> γ_init ∈ {0.1, 0.5, 1.0}, 5–10 epochs).

This re-read of the actual repo code surfaces three **load-bearing gaps**
the prior diagnosis missed and three **smaller refinements**. The P2 plan
is not refuted, but several **cheaper interventions should be tried first**
because P2 is a high-risk training-side bet on a hypothesis that the local
evidence partially weakens.

Supersedes (in part) the close-out claim at
`analyses/2026-05-01_v17g_gamma_int_boost_result.md` §"v17 inference-side
TTT path SATURATED". Local code reading shows there is at least one un-tested
inference-side lever and one un-tested checkpoint within the existing v16 run.

---

## 1. What was re-checked

Source-level reads (this turn, not from prior synthesis docs):

- [src/piano/models/motion_generator.py](src/piano/models/motion_generator.py)
  full file. Per-block IntXAttn + γ_int gate (scalar / per_head, zero-init);
  CFG dual-condition forward; MaskGIT loop in `generate()` lines 866–948.
- [src/piano/models/motion_generator_residual.py](src/piano/models/motion_generator_residual.py)
  full file. Residual transformer's per-block IntXAttn + γ_int_res
  (zero-init, per-head default); `forward_with_int` / `generate_with_int`
  thread z_int through every residual layer.
- [src/piano/inference/contact_guidance.py](src/piano/inference/contact_guidance.py)
  full file. `_scaled_gamma_int` (γ_int boost), `_decode_with_relaxed_masked_base`
  (per-step relaxed decode), `_generate_with_per_step_guidance` (re-rolled
  MaskGIT loop), `guide_with_contact` (public entry).
- [src/piano/training/decoded_contact_loss.py](src/piano/training/decoded_contact_loss.py)
  full file. `_target_trajectory_loss_canonical` (training-time geometric
  loss in **object-local frame**) with `part_margin_weight`,
  `segment_consistency_weight`, `moving_frame_extra_weight`,
  `velocity_moving_only`.
- [scripts/stage_b_generator/qual_eval.py](scripts/stage_b_generator/qual_eval.py)
  γ_int boost section (lines 666–680) + `--guidance-loss` default ("metric"
  in qual_eval, **overridden to "target" in run_v17_per_step_guidance.sh**).
- `runs/wandb_logs/wandb_history_genB_v15_alignment_guided.csv` and
  `wandb_history_genB_v16_alignment_mirror.csv` — γ_int per-epoch
  trajectory + val_contact_mean_min_dist; identified the epoch
  `best_contact.pt` was selected at and the parameter values at that epoch.
- [src/piano/training/train_generator.py](src/piano/training/train_generator.py)
  lines 1100–1265: contact-aware checkpointing config, `contact_best_key`
  default = `"mean_min_dist"`.
- [scripts/stage_b_generator/inspect_generator_ckpt.py](scripts/stage_b_generator/inspect_generator_ckpt.py)
  γ_int parsing logic (reads from saved state_dict, not wandb).

---

## 2. Pre-agent checklist outcome (per `/restart` Phase 2 §2.1)

The skill mandates a five-item local trace before any literature work.

| check | status | finding |
|---|---|---|
| Training script read end-to-end | ✓ | `target_trajectory` loss in object-local frame, with part-margin / segment-consistency / moving-boost terms |
| Inference script read end-to-end | ✓ | per-step inner loop runs in **world frame**, with `_masked_contact_l2` only — a STRICT SUBSET of training loss terms |
| Multi-checkpoint sanity check | ✓ | v16 has `best_contact.pt` (ep 70), `best_val.pt`, `final.pt` — `final.pt` has BETTER correct-part recall and same-part local error than `best_contact.pt`, but **v17 work all used `best_contact.pt`** |
| Metric site located | ✓ | val-time `mean_min_dist` ([train_generator.py:1257-1260](src/piano/training/train_generator.py#L1257-L1260)) is the *checkpoint selector*; ship metrics (correct-part recall, same-part local error) are *not* in the selector |
| Pipeline branches compared (training vs inference) | ✓ | training-time forward includes part-margin, segment-consistency, moving-boost; inference per-step does NOT |

Nothing dispatched to research agents — local trace alone surfaced enough
load-bearing findings to refine the diagnosis. Per the `/restart` skill:
"Don't dispatch literature agents while the local pipeline trace is
incomplete."

---

## 3. Findings

### F-1 — γ_int boost (v17-G) IS correctly applied to BOTH base + residual transformers (DIAGNOSIS STANDS)

[contact_guidance.py:99-140](src/piano/inference/contact_guidance.py#L99-L140) +
[qual_eval.py:666-680](scripts/stage_b_generator/qual_eval.py#L666-L680).

`--gamma-int-boost` iterates over `(transformer, res_transformer)` and
multiplies every parameter ending in `gamma_int` by the boost. Verified
against parameter naming in `motion_generator.py` (base) and
`motion_generator_residual.py` (residual) — both use the suffix
`gamma_int`. The 8 base layers (×1 scalar or ×6 per-head = 48) and 6
residual layers (×6 per-head = 36) are all scaled. **v17-G is a true
full-mechanism boost test, not a partial one.** The "boost ≥ 5
catastrophic" finding is genuine.

### F-2 — z_int IS plumbed at residual transformer (DIAGNOSIS STANDS)

[motion_generator_residual.py:107-525](src/piano/models/motion_generator_residual.py#L107-L525).

`ResidualTransformerWithInteraction` wraps the residual transformer's
inner `seqTransEncoder` with `MaskTransformerEncoderWithInteraction`.
Every residual layer has its own per-head `gamma_int_res` (zero-init).
Inference path goes through `generate_with_int` ([contact_guidance.py:493-516](src/piano/inference/contact_guidance.py#L493-L516)),
which calls `forward_with_cond_scale_with_int` with `int_kv` plumbed
into every residual layer's IntXAttn. So z_int conditioning is identical
between training and inference at the residual stage. Cross-pipeline trap
"inference drops z_int" does **not apply here**.

### F-3 — γ_int growth from v15 (0.016) to v16 (0.020) had marginal contact effect (REFINES P2 HYPOTHESIS)

From the wandb csvs:

| run | best_contact epoch | γ_int at that epoch | γ_int_res | val contact_mean_min_dist | offline 80-clip full |
|---|---:|---:|---:|---:|---:|
| v15 | 45 | 0.0161 | 0.0081 | 0.2714 m | 27.62 cm |
| v16 | 70 | 0.0204 | 0.0112 | 0.2482 m | 26.79 cm |

**γ_int grew +27 % between v15 and v16** (training distribution change:
v15 → v15+mirror-doubling). **Contact improved only −3 %** (27.62 → 26.79 cm).
correct-part recall improved more (0.168 → 0.176 raw, 0.199 on `final.pt`),
which is the right metric to look at, but the magnitude is small.

The P2 hypothesis ("if γ_int could grow more, contact alignment would
improve substantially") is partially weakened by this. Empirically,
within the 0.016 → 0.020 range, γ_int growth does have a positive sign
on alignment but small magnitude. A 25× boost from 0.020 to 0.5
(P2 γ_init=0.5 candidate) is far outside the regime that has any
empirical evidence.

The v17-G result (boost = 2 mixed, boost ≥ 5 catastrophic) gives a
direct measurement of the network's "γ_int change tolerance window" at
inference time: well below 2×. P2 asks the network to absorb a 5–25×
change via finetuning. Whether 5–10 epochs is enough for that
re-equilibration is unmeasured.

### F-4 — All v17 work was done on `best_contact.pt`; `final.pt` has measurably better alignment metrics (UN-TESTED LEVER)

From [train_generator.py:1257-1260](src/piano/training/train_generator.py#L1257-L1260):
`contact_best_key = cfg.training.contact_eval.best_key` defaults to
`"mean_min_dist"` — the legacy any-part-min-PC distance. This is the
checkpoint-selection metric for `best_contact.pt`.

But the v15/v16 ship metrics include `correct_part_recall`, `same_part_local_error`,
`moving_contact_iou` — all of which are NOT in the selector. PROGRESS notes
already flag that the available alignment-aware selectors
(`contact_alignment_contact_score`, `contact_alignment_moving_same_part_recall`)
are degenerate-early-maximum (peak at epochs 5/35 with very small values).
So they were not used. But `final.pt` (the last-epoch save) is also
available, and it's not a buggy-selector ckpt — it's just the end of
training.

PROGRESS.md row at line 106:
> Same-part local error 53.49 (bc) / 53.23 (bv) / 52.91 (final) cm vs v15's
> 55.09 / 59.92 / 54.24. On `final` ckpt, moving correct GT-part recall is
> `0.1990` — the highest non-oracle value in the project.

So `final.pt` has:
- correct-part recall 0.199 vs `best_contact.pt`'s 0.176 (+2.3 pp)
- same-part local 52.91 cm vs `best_contact.pt`'s 53.49 cm (−0.58 cm)

This is exactly the alignment metric profile the v17 series has been
chasing. **All v17 inference experiments (v17-C, D, E, F, G) ran on
`best_contact.pt`.** `final.pt` was never tested with v17-E.20 / E.50
inference recipe.

This is a textbook instance of the `/restart` skill's "checkpoint-selection
metric ≠ ship metric" trap. Cheap fix: re-run v17-E.20 + v17-E.50 on
v16 `final.pt`. Cost: ~80 + ~140 min server time, no code change.

### F-5 — Per-step inference loss is a STRICT SUBSET of training-time decoded contact loss (UN-TESTED LEVER)

The training-time `_target_trajectory_loss_canonical`
([decoded_contact_loss.py:129-294](src/piano/training/decoded_contact_loss.py#L129-L294))
includes:

| term | weight in v15/v16 | what it does |
|---|---:|---|
| `target_position` | 1.0 | per-part L2 to `contact_target_xyz` (object-local) |
| `target_velocity` | 0.5 | per-part velocity matching to target trajectory delta |
| `metric_loss` | 0.0 (off in v15/v16) | legacy any-part-min-PC distance |
| **`part_margin`** | configurable; `part_margin_m=0.08` | **wrong-part margin: penalize *other* parts that are closer to target than the GT contact part** |
| **`segment_consistency`** | configurable | keep object-local body-target offset stable across contact segment (OMOMO/CHOIS pattern) |
| `moving_frame_extra_weight` | 2.0 | doubles loss on moving-object frames |
| `velocity_moving_only` | True | velocity term only active on moving-object frames |

The inference per-step `_masked_contact_l2`
([contact_guidance.py:334-352](src/piano/inference/contact_guidance.py#L334-L352))
is JUST `(body_world - target_world)² × contact_state` — strict subset.

**This is the most likely mechanism for the "right area, wrong patch"
visual failure.** Without `part_margin`, the per-step optimizer can
satisfy the L2-to-target objective by pushing the GT contact part to
the GT target while *also* having other body parts crowded near the
contact area. The eval-time `correct_GT-part_recall` metric (which IS
part-aware, per measure_contact_alignment.py) then measures whether
the GT contact part is the closest-to-object part — and sees that
multiple parts are nearly tied, so the GT part is "right" only in some
frames. Without `segment_consistency`, the optimizer can satisfy contact
at sparse contact_state-active frames while the body drifts in non-contact
intermediate frames.

**Cheap fix:** port `part_margin` and `segment_consistency` from training
to the inference per-step loss. ~50 LOC change in
`contact_guidance.py::_generate_with_per_step_guidance`. No retraining.
Cost: a few hours dev + ~80 min server eval per ablation.

### F-6 — Per-step inner loop uses a frozen residual context approximation (UN-MEASURED DRIFT)

[contact_guidance.py:414-436](src/piano/inference/contact_guidance.py#L414-L436)
+ [contact_guidance.py:439-490](src/piano/inference/contact_guidance.py#L439-L490)
+ [contact_guidance.py:942-996](src/piano/inference/contact_guidance.py#L942-L996).

Sequence of events in `guide_with_contact` with `per_step_iters > 0`:

1. Run baseline `transformer.generate(...)` to get naive `base_ids_baseline`.
2. Run `_generate_residual_tokens(...)` once on `base_ids_baseline` to get `all_ids` (full RVQ).
3. Compute `baseline_residual_emb_sum` ONCE = `sum_{q=1..Q-1} codebook[q][all_ids[..., q]]`.
4. Re-roll MaskGIT with per-step inner loop. Inside: for each MaskGIT iter,
   run `per_step_iters` AdamW steps using `_decode_with_relaxed_masked_base`.
   That decode uses `baseline_residual_emb_sum` (frozen from step 3) +
   relaxed-soft base_logits.
5. After per-step finishes, re-run `_generate_residual_tokens(...)` on the
   NEW `base_ids_baseline` (the post-guidance base) to get fresh residuals.
6. Decode through frozen VQ.

The optimizer in step 4 is solving:
> Find base_logits that minimize loss(motion_decoded(softmax(logits) ⊙ codebook[0] + **frozen baseline_residual_emb_sum**)).

The motion that's actually saved comes from step 5–6:
> motion = vq.decoder(codebook[0][argmax(logits)] + **post-guidance residual_emb_sum**).

These two motions differ if the residual transformer's response to the
new base_ids differs from its response to the naive base_ids. The
optimizer doesn't see this drift; it converges to a logits configuration
that satisfies the wrong residual context.

**This is documented as an honest approximation in the design doc §3.1
(`analyses/2026-05-01_per_step_guidance_design.md`). The drift magnitude
is unmeasured.** If the optimizer's converged motion has loss L_opt and
the final post-residual-rerun motion has loss L_final, the gap
(L_final − L_opt) measures the approximation error.

**Cheap diagnostic:** in `guide_with_contact`, after step 5–6, re-evaluate
the same loss on the final motion and log it alongside `loss_final`. On a
few clips, compare L_opt vs L_final. If gap is small (<1 cm) the
approximation is fine; if large (>5 cm) per-step is being misled, and a
mid-loop residual-context refresh becomes worth the compute.

### F-7 — v17-E.50 metric-gaming with `loss_mode="target"` is real, not a measurement bug (DIAGNOSIS STANDS, MECHANISM CLARIFIED)

[run_v17_per_step_guidance.sh:57](scripts/stage_b_generator/run_v17_per_step_guidance.sh#L57):
`GUIDANCE_LOSS="${GUIDANCE_LOSS:-target}"` — v17 runner overrides
qual_eval's default ("metric") to "target". So v17-E uses
`_masked_contact_l2` against `contact_target_xyz_local` lifted to world.

`contact_target_xyz_local` is documented at
[contact_guidance.py:274-284](src/piano/inference/contact_guidance.py#L274-L284)
as "the closest-surface-point in object-local frame via
trimesh.proximity.closest_point" (per
`src/piano/data/pseudo_labels/extract_target.py`). So `target_world` IS
a point on the object surface.

500 AdamW steps (10 outer × 50 inner) on per-part L2 to GT surface points
with no physical-plausibility regularizer can push contact-active parts
arbitrarily close to the GT surface — closer than the GT motion itself
when run through the VQ codec (which has reconstruction error
GT_orig 13.09 cm → GT_VQ_roundtrip 18.47 cm).

The visual review's "right area, wrong patch" is consistent with this:
contact-active body parts are near the GT contact target points, but the
non-physical body distortions induced by 500 unregularized steps mean
non-contact frames and intermediate body-shape continuity are degraded.

**Implication:** v17-E.50's contact 16.50 cm is real for the
`mean_min_dist_per_frame` metric but is metric-gaming-shaped per the user's
visual review. v17-E.20 (contact 18.62 cm, just above GT roundtrip) is the
defensible ship config for paper purposes.

### F-8 — Training and inference loss frames differ (object-local vs world) but distances are equal (NOT A BUG)

Training computes distance in **object-local frame** ([decoded_contact_loss.py:185](src/piano/training/decoded_contact_loss.py#L185));
inference computes distance in **world frame**
([contact_guidance.py:1078-1091](src/piano/inference/contact_guidance.py#L1078-L1091)).
Both end up computing `||body - target||` after rigid transforms; rigid
transforms preserve distance, so the loss VALUE and gradient direction
through the rigid lift are unchanged. Not a frame-mismatch bug.

---

## 4. Refined diagnosis

The prior diagnosis ("γ_int is undertrained") **is not refuted**, but is
narrower than presented:

1. γ_int is correctly plumbed at training and inference (F-1, F-2). γ_int
   growth between v15 and v16 (0.016 → 0.020) had marginal contact effect
   (F-3). v17-G's mixed-at-2× / catastrophic-at-≥5× shows the network's
   inference-time tolerance window for γ_int change is < 2× (F-3).
2. The "saturation" claim at the end of the v17 series (`v17 inference-side
   path SATURATED, five levers tested`) is **incomplete**. At least two
   inference-side levers were not tested:
   - **Inference per-step loss = strict subset of training loss** (F-5).
     `part_margin` and `segment_consistency` from the v15/v16 training
     objective are not active during per-step. Adding them is a small code
     change with no retraining.
   - **`final.pt` was never evaluated under v17-E** (F-4). v16 `final.pt`
     has higher correct-part recall (0.199) and lower same-part local
     error (52.91 cm) than `best_contact.pt`. v17-E.20 / E.50 on
     `final.pt` is one server run away.
3. The frozen-residual-context approximation in per-step (F-6) is unmeasured.
   If the drift is large, it caps the per-step ceiling regardless of γ_int.

---

## 5. Decision tree (revised)

Order from cheapest to most expensive. Each step is a checkpoint on whether
the next step is justified.

### B1 — Re-eval v17-E.20 + v17-E.50 on v16 `final.pt` (~3.5 h server)

Hypothesis: `final.pt` has +2.3 pp correct-part / −0.58 cm same-part
local advantage over `best_contact.pt` (PROGRESS measured). Per-step on
top should preserve that delta.

Triggers:
- Predicted: contact 18 cm, correct-part 0.27, local 41 cm (≈ E.20 on
  best_contact + the final.pt baseline delta).
- If correct-part > 0.22 and local < 42 cm without metric gaming →
  `final.pt + v17-E.20` is the new ship config; close out.
- If gain is < 1 pp correct-part, `best_contact.pt` was good enough;
  proceed to B2.

Cost: zero code; just re-run `run_v17_per_step_guidance.sh` with
`SOURCE_RUN_DIR=...generator_v16_alignment_mirror` and `CKPTS=final`.

### B2 — Add part_margin + segment_consistency to per-step inner loss (~1 day code + 4 h server)

Hypothesis: training-time `part_margin` (wrong-part penalty) and
`segment_consistency` (object-local offset stability) are the
inference-time terms missing to fix "right area, wrong patch" visual
failure. Their training-time gradient evidence is documented in
[stageB_compact.md] as actively used in v15/v16.

Code change scope: ~50 LOC in
`contact_guidance.py::_generate_with_per_step_guidance` to add the two
terms with same formulation as
`decoded_contact_loss.py::_target_trajectory_loss_canonical` lines
204–248 (port the two if-blocks). New CLI flags:
`--per-step-part-margin-weight`, `--per-step-segment-consistency-weight`.

Triggers:
- Run on the same ckpt that B1 selected.
- Sweep `part_margin_weight ∈ {0.5, 1.0, 2.0}` first.
- If correct-part recall jumps ≥ 5 pp at any setting → ship `v17-H`
  (per-step + part-margin) as the new default.
- If no improvement at any setting → port to ablation log; proceed to B3.

### B3 — Measure residual-context drift (F-6) (~half day code + 4 h server)

Hypothesis: the per-step optimizer's converged L_opt and the actual
post-residual-rerun L_final differ enough to cap per-step's ceiling.

Code change: in `guide_with_contact`, after the residual rerun, recompute
the same geometric loss on the final motion (`motion_norm * std + mean →
recover_from_ric → world-lift → masked_L2`). Log alongside `loss_final`
in `info` dict. ~30 LOC.

Triggers:
- |L_final − L_opt| < 1 cm → approximation is fine; per-step is genuine
  saturation. Move to B4 / pivot.
- |L_final − L_opt| > 5 cm → mid-loop refresh of `baseline_residual_emb_sum`
  is worth implementing (`per_step_residual_refresh_every=N` MaskGIT
  steps; cost ~5× per-step compute).

### B4 — P2 (γ_int re-init + Stage B finetune), but with smaller γ_init and larger sweep (~1 day code + 12 h server)

If B1+B2+B3 don't close the correct-part gap, P2 is justified — but the
existing P2 plan (γ_init ∈ {0.1, 0.5, 1.0}) is too aggressive given the
v17-G evidence that the network's tolerance window is < 2× at inference.

Revised γ_init candidates: `{0.05, 0.1, 0.2}`. The 0.05 candidate (2.5×
the trained 0.020 plateau) is the lowest-risk smoothing of the current
state; if it improves, scaling up to 0.1 / 0.2 / 0.5 becomes justified
*incrementally*. The v17g doc's γ_init=1.0 candidate should be
deprioritized until 0.05 / 0.1 / 0.2 give signal.

Decision rule unchanged from
`analyses/2026-05-01_v17g_gamma_int_boost_result.md` §"Decision tree".

### B5 — Pivot to OMOMO-style explicit `contact_target` as input (~1 week)

Final fallback. Architecture change using existing predictor output channel
(Stage A already produces `contact_target_xyz`). This is what v17g doc
flags as the negative-result fallback.

---

## 6. Ranked impact estimate

| branch | dev cost | server cost | predicted Δ correct-part | confidence |
|---|---:|---:|---:|---|
| B1 final.pt re-eval | 0 | ~3.5 h | +2 pp | HIGH (existing PROGRESS data) |
| B2 part_margin + seg_consistency in per-step | ~1 day | ~4 h | +5–10 pp | MEDIUM (analogous to training-time evidence; mechanism direct) |
| B3 residual drift diagnostic | ~0.5 day | ~4 h | 0 directly; unblocks ceiling check | HIGH |
| B4 γ_init=0.05 finetune | ~1 day | ~3 h | +1–3 pp | LOW (v15→v16 25% γ growth → 1 pp correct-part) |
| B4 γ_init=0.5 finetune | ~1 day | ~3 h | wide variance — could be +10 or −catastrophic | LOW (extrapolation from v17-G inference behaviour at 5×) |

Ordering: **do B1 + B2 first** before any P2 commitment. B1 is zero-code.
B2 is small code. Both have higher predicted impact and lower risk than
the existing P2 plan.

---

## 7. References

### Internal

- Source files (with line ranges given as anchored citations in §3 above):
  [src/piano/models/motion_generator.py](src/piano/models/motion_generator.py),
  [src/piano/models/motion_generator_residual.py](src/piano/models/motion_generator_residual.py),
  [src/piano/inference/contact_guidance.py](src/piano/inference/contact_guidance.py),
  [src/piano/training/decoded_contact_loss.py](src/piano/training/decoded_contact_loss.py),
  [src/piano/training/train_generator.py](src/piano/training/train_generator.py),
  [scripts/stage_b_generator/qual_eval.py](scripts/stage_b_generator/qual_eval.py),
  [scripts/stage_b_generator/run_v17_per_step_guidance.sh](scripts/stage_b_generator/run_v17_per_step_guidance.sh),
  [scripts/stage_b_generator/inspect_generator_ckpt.py](scripts/stage_b_generator/inspect_generator_ckpt.py).
- Wandb csvs: `runs/wandb_logs/wandb_history_genB_v15_alignment_guided.csv`,
  `runs/wandb_logs/wandb_history_genB_v16_alignment_mirror.csv` —
  per-epoch γ_int / γ_int_res / val_contact_mean_min_dist trajectory.
- Prior synthesis (this re-diagnosis supersedes their "saturated" claim):
  [analyses/2026-05-01_v17g_gamma_int_boost_result.md](analyses/2026-05-01_v17g_gamma_int_boost_result.md),
  [analyses/2026-05-01_v17f_gumbel_result_and_p1_plan.md](analyses/2026-05-01_v17f_gumbel_result_and_p1_plan.md),
  [analyses/2026-05-01_v17_per_step_result.md](analyses/2026-05-01_v17_per_step_result.md),
  [analyses/2026-05-01_per_step_guidance_design.md](analyses/2026-05-01_per_step_guidance_design.md).
- Project state pinned at restart: [PROGRESS.md](PROGRESS.md),
  [PLAN.md](PLAN.md), [analyses/stageB_compact.md](analyses/stageB_compact.md).

### External (referenced in the prior series but not load-bearing for this re-diagnosis)

- Pinyoanuntapong, E. et al. *MaskControl: Spatially-Conditioned Generation
  of Discrete Motion via Logit Optimization.* **ICCV 2025**.
  arXiv:2410.10780. Source: `exitudio/ControlMM`.
- Karunratanakul, K. et al. *Optimizing Diffusion Noise Can Serve As
  Universal Motion Priors.* **CVPR 2024**. arXiv:2312.11994.
- Guo, C. et al. *MoMask: Generative Masked Modeling of 3D Human Motions.*
  **CVPR 2024**. arXiv:2312.00063.
- Li, Y. et al. *OMOMO: Object Motion Guided Human Motion Synthesis.*
  **SIGGRAPH Asia 2023**. arXiv:2309.16237. (Hand-position intermediate
  target — fallback in B5.)
