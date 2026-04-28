# PIANO Current Recommendation

Compact recommendation memo as of 2026-04-29.

## Executive Position

Do not continue tuning decoded-contact loss weight or checkpoint selection as
the main Stage B strategy. v0.12 shows the training surrogate can be strengthened
without moving generated contact below the same roughly `32 cm` band.

The next correct move is no-retrain diagnosis:

1. K-sample oracle.
2. Soft-hard gap diagnostic.
3. RVQ mixed oracle.
4. Subset-specific codebook audit.

Current framing: this is not simply "contact loss too weak." The likely failure
is a mismatch between the soft decoded auxiliary path used in training and the
hard sample-time RVQ path used in evaluation.

## Evidence

v12 matched 80-clip eval:

| checkpoint | full |
|---|---:|
| w02 best_val | 31.82 |
| w08 final | 32.17 |
| w03 best_contact | 32.39 |
| w02 final | 32.51 |

Top checkpoints are too close to make checkpoint selection a path to GT
roundtrip (`18.47 cm`).

Decoded-contact weight sweep:

| weight | decoded grad median | final decoded loss | best full |
|---:|---:|---:|---:|
| 0.20 | 5.57% | 0.1558 | 31.82 |
| 0.30 | 7.97% | 0.1340 | 32.39 |
| 0.50 | 13.02% | 0.1128 | 32.87 |
| 0.80 | 19.27% | 0.0988 | 32.17 |

The surrogate behaves as intended, but sample-time contact does not follow.

## What Already Worked

- MoMask encoder normalization fix: repaired token/body collapse.
- Contact-distance metric: exposed the real failure mode.
- Per-head gamma: gave the main early architecture improvement.
- Decoded contact auxiliary loss: made `z_int` matter for contact.
- Full-RVQ decoded-contact path: partial improvement on 20 clips.

## What Is Not Worth Repeating Blindly

- More CE training.
- Mirror augmentation.
- Trainable-copy InterControl variant without new diagnosis.
- Base-logit-only guidance as the main solution.
- Residual `z_int` conditioning alone.
- Larger decoded-contact weights.
- Picking a different v12 checkpoint and expecting a large jump.

## Recommended Next Implementation

Add diagnostic scripts under `scripts/stage_b_generator/` and reuse library
helpers from `src/piano/`:

- use existing eval sampling so the 80-clip set is matched;
- use upstream MoMask/VQ decode and `recover_from_ric` paths;
- use existing contact-distance code rather than reimplementing geometry;
- output small `summary.json` files, not large generated arrays unless needed.

First script landed:

- `scripts/stage_b_generator/k_sample_oracle.py`

Decision rules:

- K-sample oracle succeeds: build reranking/guidance around the existing
  distribution.
- K-sample oracle fails: revise model/training distribution.
- Soft-hard gap is large: move decoded contact closer to hard sampling with
  ST-Gumbel/DES-style consistency or full-RVQ logits/embedding optimization.
- RVQ mixed oracle identifies base bottleneck: focus MaskTransformer/base
  conditioning.
- RVQ mixed oracle identifies residual bottleneck: focus residual/full-RVQ
  conditioning and loss.
- Codebook audit shows subset roundtrip failure: treat representation as a
  subset-specific bottleneck, especially IMHD.
