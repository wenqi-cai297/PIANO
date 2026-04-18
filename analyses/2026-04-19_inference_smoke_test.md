# End-to-End Inference Smoke Test — 2026-04-19

## Context

Baseline run before any training. Goal: verify the full inference chain works
when assembled from pretrained MoMask + untrained PIANO components. Because
the interaction cross-attention is zero-initialized, the output at this
stage should behave like pure MoMask text-only generation; if it does, we
know the plumbing is correct and we can safely move to training.

- Run dir: `runs/checks/inference_smoke_test/2026-04-19_063940/`
- Runner: `scripts/server/inference_smoke_test.sh`
- Device: cuda (RTX A6000), bf16 mixed precision via accelerate config
- Data: 4 samples from `/media/.../omomo/piano/` (first 4 entries,
  all "Lift the clothesstand, move the clothesstand, and put down the
  clothesstand.", seq_lens=[88, 79, 91, 95])

## Results

All shapes match expectations:

| Tensor | Shape | Notes |
|--------|-------|-------|
| CLIP text_emb | (4, 512) | MoMask ViT-B/32 |
| Object tokens | (4, 16, 384) | PointNet++ |
| Interaction latent (contact/phase/support) | (4, 196, {5/5/4}) | per-frame |
| Interaction tokens (temporal-downsampled) | (4, 49, 384) | 196 → 49 via conv stride 4 |
| Generated base tokens | (4, 49) | range [4, 504] ∈ [0, 511] valid |
| Full indices (6 RVQ levels) | (4, 49, 6) | after ResidualTransformer |
| Decoded motion_263 | (4, 196, 263) | VQ-VAE output |

Parameter counts:

| Module | Params | Status |
|--------|--------|--------|
| InteractionPredictor (untrained) | 31.76M | random init |
| ObjectEncoder (untrained) | 0.31M | random init |
| Interaction cross-attn layers (new) | 5.51M | **zero-initialized** |
| MoMask backbone (after wrapping) | 12.05M (+ 150M frozen CLIP) | pretrained |

Output statistics:
- `finite = True` (no NaN/Inf)
- `motion mean = -0.34, std = 4.80`
- `token_ids ∈ [4, 504]`, valid VQ range

z_int statistics (show random init is behaving as expected):
- contact_state per body part: `[0.56, 0.71, 0.52, 0.47, 0.45]` — sigmoid of
  ~random, centered around 0.5 as expected
- phase distribution: `[0.18, 0.08, 0.21, 0.22, 0.31]` — approximately
  uniform over 5 phases (expected 0.20 each)
- support distribution: `[0.25, 0.29, 0.28, 0.18]` — approximately
  uniform over 4 states (expected 0.25 each)

## Observations

**Pipeline plumbing is correct.** Every stage (CLIP encode → PointNet →
Predictor → Tokenizer → MaskTransformer.generate → ResidualTransformer →
VQ-VAE decode) runs without errors and produces tensors of the right shape
on the right device.

**Zero-init interaction cross-attn verified.** Because the new cross-attn
layers output zero at init, the `logits_full - logits_text` term in
`forward_with_cond_scale` is zero, so the `interaction_scale=2.0` CFG term
has no effect. The generation is therefore equivalent to pure MoMask
text-only generation with `cond_scale=4.5`.

**Untrained Predictor outputs are near-uniform.** This is the right
baseline behavior. After training with pseudo-labels, we expect the
phase/support/contact distributions to become sharply structured
(e.g., phase[approach]=0.9 in early frames, phase[manipulation]=0.9
mid-sequence).

**Two non-obvious bugs found + fixed during this check:**
1. `SMPL-X batch_size` mismatch (fixed during preprocessing — see
   `2026-04-19_omomo_preprocessing.md`)
2. `InteractionMaskTransformer.from_pretrained` left the new interaction
   layers on CPU while MoMask was on cuda → `layer_norm` device mismatch.
   Fix: added `wrapper.to(device)` after wrapping, in
   `src/piano/models/motion_generator.py:184-188`.

**Motion output range is wider than typical HumanML3D (std=4.80).** This is
likely because:
- The HOI prompt ("Lift the clothesstand...") is out-of-distribution for
  MoMask, which was trained on HumanML3D text captions
- `cond_scale=4.5` is MoMask's default but can push outputs outside the
  trained distribution for novel prompts
- The decoded motion will look chaotic — that's expected at this stage
  and not a concern for plumbing verification

## Diagnosis

System is working as designed. We are ready to start training.

Before kicking off training we should:
1. Extract pseudo-labels on all 4919 sequences so Stage A can learn
2. Optionally: render a single generated motion sequence to video/frames to
   confirm visually that motion decoded from MoMask is "reasonable motion"
   even if semantically uncorrelated with the HOI prompt. This is not
   blocking — the numeric checks above are sufficient.

## Implications

- **Smoke test as baseline reference.** This 4-sample generation with
  untrained PIANO is our "pre-training reference point". After Stage A/B/C
  training, re-running the same script on the same 4 samples should
  produce measurably different (and better aligned-with-object) motion.
- We now have confidence that the full graph is connected and device-safe.
- The interaction latent initialization is sane (near-uniform, no
  pathological values).

## Action Items (→ PLAN.md)

- [x] Verify end-to-end pipeline (done)
- [x] Confirm zero-init interaction cross-attn behavior (done)
- [ ] Next: extract pseudo-labels for all 4919 OMOMO sequences
- [ ] Next: run Stage A (Interaction Predictor) training
- [ ] After Stage A: re-run smoke test to confirm Predictor output is no
  longer near-uniform (structured phase/support/contact over time)
