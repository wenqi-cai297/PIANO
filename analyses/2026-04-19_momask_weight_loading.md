# MoMask Weight Loading Verification — 2026-04-19

## Context

First-time setup on the A6000 GPU server. Before any training, we needed to
confirm the critical assumption: our `momask_adapter.py` can load MoMask's
three pretrained HumanML3D checkpoints cleanly via the original MoMask classes.

- Server: single RTX A6000 (48GB), CUDA 12.2 driver, PyTorch 2.5.1+cu121
- MoMask checkpoints at `checkpoints/momask/t2m/`
- Adapter at `src/piano/models/backbones/momask_adapter.py`
- Test runner: `scripts/server/check_momask_weights.sh`

## Results

| Checkpoint | Params | Status |
|------------|--------|--------|
| VQ-VAE (`net_best_fid.tar`) | 19.4M | Clean load |
| MaskTransformer (`latest.tar`) | 163.3M (incl. CLIP ~150M) | Clean load |
| ResidualTransformer (`net_best_fid.tar`) | 164.6M (incl. CLIP ~150M) | Clean load |

No `[WARN] Missing non-CLIP keys` or `[WARN] Unexpected keys` messages after
fixes. CLIP ViT-B/32 downloaded automatically on first run (~300MB).

## Observations

### Two non-trivial issues found during debugging

**Issue 1: VQ-VAE required `mu` in opt namespace**

MoMask's `QuantizeEMAReset.__init__` reads `self.mu = args.mu` (EMA decay for
codebook). Our initial `build_momask_opt` didn't set it.

Fix: add `mu=0.99` default (matching `options/vq_option.py` default) to
`build_momask_opt()` in `momask_adapter.py`.

**Issue 2: ResidualTransformer required `share_weight=True`**

Without it, three keys in the checkpoint had no corresponding model parameters:
- `output_proj_weight`
- `output_proj_bias`
- `token_embed_weight`

Root cause: the checkpoint folder name `tres_..._sw` encodes `share_weight=True`.
Our adapter defaulted to `False`, which picks the wrong branch in
`ResidualTransformer.__init__` (line 682 vs line 689 of MoMask's
`transformer.py`), creating different weight tensors with different names.

Fix: default `share_weight=True` in `load_momask_residual_transformer()`.

### Other setup notes

- Driver 535.171 requires PyTorch cu121 (cu124 fails with "driver too old")
- RTX A6000 supports bf16, set in `~/.cache/huggingface/accelerate/default_config.yaml`
- OpenAI CLIP (not HuggingFace) is needed: `pip install ftfy regex git+https://github.com/openai/CLIP.git`
- PyTorch3D not actually used — skipped installation (pip wheel doesn't exist)
- MoMask is a flat script repo → cloned into `src/piano/models/backbones/momask/`
  and added to `sys.path` by the adapter at import time. No `pip install` needed
  for MoMask itself.

## Diagnosis

The fixes are structural knowledge about MoMask's internal conventions that
are not documented in its README. Discoverable only by:
1. Reading the actual `options/vq_option.py` for defaults
2. Decoding the checkpoint folder naming convention (`_sw` suffix)
3. Running load + reading `[WARN] Missing keys` carefully

## Implications

- The pretrained-weight loading path is now 100% verified. Stage B/C finetune
  is not blocked on this.
- Server environment (driver/CUDA/PyTorch/CLIP) is correctly matched and usable.
- We can move on to data prep with confidence.

## Action Items (→ PLAN.md)

- [x] Document fixes in the adapter with comments referencing MoMask's defaults
- [x] Update PROGRESS.md to mark this as verified
- [ ] Next: prepare OMOMO dataset
