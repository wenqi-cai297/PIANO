# PIANO Current Recommendation

Compact recommendation memo as of 2026-04-29.

## Executive Position

Do not continue tuning decoded-contact loss weight or checkpoint selection as
the main Stage B strategy. v0.12 shows the training surrogate can be
strengthened without moving one-shot generated contact below the same roughly
`32 cm` band.

The K-sample oracle changes the active plan: the current generator distribution
already contains good-contact samples. K=16 best-of-K reaches `17.93 cm`, close
to GT VQ roundtrip `18.47 cm`. The next correct move is contact-aware
reranking/guidance, with visual review to make sure the metric-selected samples
are usable motions.

Current framing: this is not simply "contact loss too weak" and not primarily
"the model cannot generate contact." The failure is that ordinary sampling does
not reliably choose the good contact modes.

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

K=16 oracle on v12 w02 best_val:

| metric | value |
|---|---:|
| single-sample mean | 32.22 cm |
| K=16 sample mean | 31.64 cm |
| K=16 best-of-K mean | 17.93 cm |
| K=16 best-of-K median | 14.50 cm |
| best under 22 cm | 70% |
| best under 25 cm | 80% |

Per-subset K=16 best-of-K:

| subset | best-of-K |
|---|---:|
| chairs | 8.44 cm |
| imhd | 29.38 cm |
| neuraldome | 21.66 cm |
| omomo_correct_v2 | 12.23 cm |

This is strong evidence for a reranking/guidance branch. IMHD still needs
targeted analysis.

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

Build the next Stage B path under `scripts/stage_b_generator/` and reuse
library helpers from `src/piano/`:

- use existing eval sampling so the 80-clip set is matched;
- use upstream MoMask/VQ decode and `recover_from_ric` paths;
- use existing contact-distance code rather than reimplementing geometry;
- output small `summary.json` files, not large generated arrays unless needed.

Current script:

- `scripts/stage_b_generator/k_sample_oracle.py`

Decision rules:

- K-sample oracle succeeded: build reranking/guidance around the existing
  distribution.
- Visual review passes: make contact-aware best-of-K reranking the Stage B
  baseline.
- Visual review fails: diagnose metric loopholes, then use soft-hard/RVQ mixed
  diagnostics before more training.
- Soft-hard gap is large: move decoded contact closer to hard sampling with
  ST-Gumbel/DES-style consistency or full-RVQ logits/embedding optimization.
- RVQ mixed oracle identifies base bottleneck: focus MaskTransformer/base
  conditioning.
- RVQ mixed oracle identifies residual bottleneck: focus residual/full-RVQ
  conditioning and loss.
- Codebook audit shows subset roundtrip failure: treat representation as a
  subset-specific bottleneck, especially IMHD.
