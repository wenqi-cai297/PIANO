# 2026-05-01 — Per-step decoded-geometric guidance (v17)

Source-of-truth design doc for the per-step MaskGIT guidance branch
implemented in this commit. Supersedes the rough description in
`stageB_compact.md` "MaskControl-style guidance" lines.

## 1. Why this branch

v16 mirror-doubled training landed a real but small gain over v15 (raw
contact 26.79 cm vs 27.62 cm; correct-part recall 0.18 vs 0.17;
same-part local error 53.49 cm vs 55.09 cm) and is still 8–9 cm worse
than the v14 K=16 distance oracle (17.60 cm) and ~13 cm worse than the
v14 K=64 alignment oracle on local error (40.30 cm). The v14 K=64
candidate-capacity table (analyses/stageB_compact.md, "v14 K=64
alignment-aware oracle") shows the per-clip best primary alignment
across 64 candidates is still 37.0 cm on average and only 9 % of clips
have any K=64 candidate with moving same-part recall ≥ 0.5. So the
distribution itself is missing GT-aligned manipulation modes; pure
selection over already-sampled candidates is near exhausted.

The current `--guidance-layers full_rvq` path optimises the full RVQ
stack only **after the MaskGIT loop has fully committed**. Per the
docstring at `src/piano/inference/contact_guidance.py:50-72`, this is a
deliberate scope reduction of the canonical MaskControl recipe, which
optimises both per MaskGIT iteration (`each_iter≈100`) and post-hoc
(`iter_last≈600`). PIANO has only the post-hoc half. The per-step half
is what lets the geometric gradient steer the **commit decision** —
which token to commit, in what order — instead of only repairing
already-committed tokens at the end.

This branch closes that gap as a no-retraining inference-time addition.
It runs on v14/v15/v16 checkpoints unchanged.

## 2. Reference: MaskControl / ControlMM ICCV 2025

Pinyoanuntapong, E. et al. *MaskControl: Spatially-Conditioned Generation
of Discrete Motion via Logit Optimization.* **ICCV 2025**. arXiv:2410.10780.
Source: `exitudio/ControlMM@models/mask_transformer/control_transformer.py`,
function `generate_with_control`, lines ~750-820 (per-iter test-time
training, "TTT") and ~890-920 (final-stage TTT).

The published recipe per iteration:

1. Run masked transformer to get logits at the current MaskGIT step.
2. Treat logits as `nn.Parameter`. Run `each_iter=100` AdamW
   (betas=(0.5, 0.9), lr=6e-2, weight_decay=1e-6) inner steps.
3. Each inner step: differentiable decode (relaxed expectation over
   codebook) → motion → spatial-control loss → `loss.backward()` →
   optimizer.step.
4. Use the *optimised* logits for the commit decision (top-k filter +
   sample + score-update).

This pattern cited and re-verified independently by `MotionLCM`
(`Dai-Wenxun/MotionLCM@mld/models/modeltype/mld.py`,
`Adam([current_latents.requires_grad_(True)], lr=...)`).

## 3. PIANO-specific adaptations

### 3.1 Multi-quantizer residual

MaskControl operates on a single-quantizer base layer. PIANO's MoMask
backbone has Q=6 RVQ layers (1 base + 5 residual). The residual
transformer is autoregressive over the 5 residual layers and is run
*after* MaskGIT base sampling completes.

During per-step guidance the residual codes do not yet exist. Two
options:

- (a) Drop residuals (treat them as code 0 across all 5 layers). Biased
  — the residual codebook entries are not the zero vector.
- (b) Use **frozen residual embeddings from a baseline pass** as the
  residual context approximation.

We use (b). Cost: one extra MaskGIT + one residual transformer pass
before the guided generation, ≈ 13 transformer forwards. Same pattern
as the existing post-hoc code which already runs a baseline pass at
`contact_guidance.py:490-507`.

Rationale: (b) gives a geometric loss whose absolute value matches the
final motion's geometry up to `Δ_residual` between baseline and
post-guidance residuals, which is small on already-converged generators.
We re-run the residual transformer on the final base_ids after the
guided MaskGIT loop, so the saved motion uses fresh residuals.

### 3.2 Relaxed-decode site

At MaskGIT step `t` with current ids `{committed: hard token, masked:
mask_id}`:

```
hard_base_emb[B,S,d]   = codebook[0, ids_committed]               # for already-committed
soft_base_emb[B,S,d]   = softmax(logits_param/T, dim=-1) @ codebook[0]
relaxed_base[B,S,d]    = where(is_masked, soft_base_emb, hard_base_emb)
x_emb[B,S,d]           = relaxed_base + baseline_residual_emb_sum     # frozen residuals
motion_norm[B,T,263]   = vq.decoder(x_emb.permute(0,2,1))
motion[B,T,263]        = motion_norm * std + mean
joints_canon[B,T,22,3] = recover_from_ric(motion, 22)              # MoMask upstream
joints_world           = R_y(angle) @ joints_canon + T_xz          # source-clip anchor
body[B,T,5,3]          = joints_world[:, :, [20, 21, 10, 11, 0]]
loss = masked_L2(body, contact_target_world) under contact_state    # same as final-stage loss
```

Strict MaskGIT semantics: only masked positions' logits matter for the
final ids. Committed positions are fixed. So we optimise only the
masked-position logits — gradient on committed-position logits does
nothing. Implementation gates this via `is_masked` indexing.

### 3.3 Hyperparameters (first prototype)

| param | value | rationale |
|---|---|---|
| `per_step_iters` | 10 | ~10× smaller than MaskControl's 100. Conservative budget for first ablation; sweep up if signal is positive but small. |
| `per_step_lr` | 6e-2 | Match MaskControl. Same as existing post-hoc `guidance_lr`. |
| `per_step_temperature` | 1.0 | Softmax-T in the relaxed decode. 1.0 = unbiased expectation. |
| `per_step_start_step` | 0 | Active from MaskGIT step 0. MaskControl applies from step 0; PIANO matches. |
| `per_step_init_from_logits` | True | Optimise the transformer's actual logits, not a one-hot. The post-hoc code uses one-hot init, but per-step has access to the model's logits directly which is the canonical MaskControl init. |
| `weight_decay` | 1e-6 | Match MaskControl AdamW config. |
| `betas` | (0.5, 0.9) | Match MaskControl. |

10 outer × 10 inner = 100 inner steps total per clip. Per inner step =
1 vq.decoder forward + 1 recover_from_ric + 1 lift + 1 L2 → ≈ 5 ms on
A6000 bf16 → ≈ 0.5 s extra per clip beyond baseline generation. 80 clips
× 0.5 s = 40 s extra wall-time per condition. Acceptable.

### 3.4 Compatibility with existing post-hoc guidance

The branch is **additive**. New CLI:

```
--per-step-iters N           # 0 disables (back-compat); >0 enables per-step
--per-step-lr 6e-2           # AdamW lr for per-step inner loop
--per-step-temperature 1.0   # softmax T for relaxed decode
--per-step-start-step 0      # MaskGIT step idx to start guiding from
```

Existing `--guidance-steps`, `--guidance-layers full_rvq`, etc. continue
to work and can stack with per-step guidance:

| `--per-step-iters` | `--guidance-steps` | behaviour |
|---:|---:|---|
| 0 | 0 | raw baseline generation |
| 0 | 30 | current v16 `full_guided` (post-hoc only) |
| 10 | 0 | **v17 main ablation**: per-step only |
| 10 | 30 | per-step + post-hoc stacked (canonical MaskControl) |

## 4. Ablation plan (v17)

Run on the existing v16 (or v14) `best_contact.pt` ckpt, no retraining:

| run | per-step | post-hoc | what it tests |
|---|---:|---:|---|
| v17-A baseline | 0 | 0 | sanity / re-baseline raw generation |
| v17-B post-hoc only | 0 | 30 | reproduce v16 `full_guided` |
| v17-C per-step only | 10 | 0 | **does per-step alone move contact?** |
| v17-D stacked | 10 | 30 | full MaskControl recipe; ceiling check |
| v17-E budget sweep | {20, 50, 100} | 0 | does more per-step iters help further? |

Decision rule:

- If **v17-C beats v17-B** on contact distance + correct-part recall
  + same-part local error → per-step guidance is the right lever.
  Continue with v17-D (stack with post-hoc) and budget sweep (v17-E).
- If **v17-C ≈ v17-B and v17-D ≈ v17-B** → the geometric gradient
  saturates against PIANO's residual stack regardless of when applied.
  Pivot to OMOMO-style hand-position intermediate target (project plan
  branch (2)).
- If **v17-C is worse than v17-B** → likely an instability (init
  misuse, residual approximation drifting too far). Diagnose with the
  per-clip loss trace before declaring per-step dead.

Success threshold (rough): match v14 K=16 distance oracle (17.60 cm)
within ≤ 5 cm raw, with correct-part recall ≥ 0.22 and same-part local
error ≤ 48 cm. This is "definitively closes most of the K=16 oracle
gap" but does not require matching K=64 alignment.

## 5. Risks / failure modes

1. **Residual approximation drift**. Frozen baseline residuals may not
   match the post-guidance base distribution. Mitigation: `--per-step-
   refresh-residual N` (deferred, not in v17 first cut) to re-run the
   residual transformer every N MaskGIT steps. For the first cut, we
   accept the drift and rely on the final residual rerun on the final
   base_ids to absorb it.
2. **Logit init-scale mismatch**. The post-hoc code uses one-hot ×
   `init_logit_scale=3.0` because softmax over a 512-vocab one-hot is
   too peaked at default init_scale=10 (`p(argmax) ≈ 0.99996`,
   per-clip notes 2026-04-28). Per-step uses the transformer's actual
   logits, which are typically lower-magnitude and softer. We expect
   this to behave like MaskControl's "init from last-iter logits"
   pattern at `contact_guidance.py:590-595`. If first runs show
   p(argmax) ≈ 1 making optimisation flat, add `--per-step-init-
   temperature` to scale logits before optimisation.
3. **Gumbel/RNG drift**. Existing `--guidance-residual-seed` covers the
   final residual rerun. Per-step base sampling uses `gumbel_sample`
   inside the MaskGIT loop. If different seeds across `v17-C` runs
   produce > 1 cm contact variance, plumb `per_step_base_seed` through
   the same way.
4. **Geometry-loss compute scales linearly with T**. Long clips (T=196
   frames) → ≈ 200 × 5 × 1024 × 3 floats per loss eval. For the inner
   loop that's the dominant cost. Acceptable on A6000.
5. **Early-step relaxed decode is meaningless**. At MaskGIT step 0, all
   positions are masked, so `motion = vq.decoder(softmax(logits) @
   codebook[0] + frozen_residuals)` is the model's "average prior
   motion", not an actual sample. Optimising against the geometric loss
   here pushes logits toward a single mode that is geometrically valid
   *under the residual approximation*, which can be an OK warm start
   or a dead-end attractor. Mitigation lever: `--per-step-start-step`
   to skip early steps. Defer until first results show whether early
   steps help or hurt.

## 6. Implementation map (this commit)

| file | change |
|---|---|
| `src/piano/inference/contact_guidance.py` | + `_decode_relaxed_with_baseline_residual_emb` helper. + `generate_with_per_step_guidance` function (re-rolls the InteractionMaskTransformer.generate loop with per-step inner optimization). Modify `guide_with_contact` to accept `per_step_iters` + related params and route accordingly. |
| `scripts/stage_b_generator/qual_eval.py` | + `--per-step-iters`, `--per-step-lr`, `--per-step-temperature`, `--per-step-start-step` CLI. Pass-through to `guide_with_contact`. Extend `guidance_trace.json` schema. |
| `scripts/stage_b_generator/run_v17_per_step_guidance.sh` | New runner. `TRAIN=0`, points at v16 `best_contact.pt`, `PER_STEP_ITERS=10`, `GUIDANCE_STEPS=0` (v17-C config). |
| `tests/test_contact_guidance_per_step.py` | New CPU-friendly shape/finite-loss test for `generate_with_per_step_guidance`. |
| `PROGRESS.md`, `PLAN.md`, `ANALYSIS.md`, `stageB_compact.md`, `restart_prompt.md` | Note v17 implementation landed, server run pending. |

The base/full-RVQ post-hoc path (`guide_with_contact` minus
`per_step_iters`) is unchanged. v15/v16 reproducibility is preserved.

## 7. References

- Pinyoanuntapong, E. et al. *MaskControl: Spatially-Conditioned
  Generation of Discrete Motion via Logit Optimization.* **ICCV 2025**.
  arXiv:2410.10780. Source code at `exitudio/ControlMM`.
  - `models/mask_transformer/control_transformer.py::generate_with_control`
    lines ~750-820 (per-iter TTT) and ~890-920 (final-stage TTT).
- Karunratanakul, K. et al. *Optimizing Diffusion Noise Can Serve As
  Universal Motion Priors.* **CVPR 2024**. arXiv:2312.11994.
  (Independent verification of the relaxed-decode + AdamW-on-latent
  pattern in masked-token motion control.)
- Dai, W. et al. *MotionLCM: Real-time Controllable Motion Generation
  via Latent Consistency Model.* **ECCV 2024**. arXiv:2404.19759.
  Source: `Dai-Wenxun/MotionLCM@mld/models/modeltype/mld.py` —
  identical AdamW-on-latents pattern.
- Guo, C. et al. *MoMask: Generative Masked Modeling of 3D Human
  Motions.* **CVPR 2024**. arXiv:2312.00063. PIANO's MaskGIT base
  + residual RVQ backbone.
- Internal: `analyses/stageB_compact.md` v14 K=64 alignment oracle
  capacity table — the empirical motivation for needing a
  distribution-shaping mechanism instead of more reranking.
- Internal: `analyses/2026-04-28_v0_3_delta_retrain_and_v0_5_contact.md`
  — the CE/contact decoupling result that motivated the original B3
  inference-time guidance recipe.
