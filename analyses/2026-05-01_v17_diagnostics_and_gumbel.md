# 2026-05-01 — v17 follow-up: γ_int audit + Gumbel-noise addition

After the v17-D / v17-E.20 / v17-E.50 sweep landed (per
`analyses/2026-05-01_v17_per_step_result.md`), three follow-up items
were prompted by visual review + a project-direction discussion with
the user:

1. **Visual review** of v17-E.20 / v17-E.50 (done by user).
   - v17-E.50 visibly better than v17-E.20 and v16 raw on contact
     placement, but body parts are at the *correct broad area* of the
     object yet at the *wrong patch* (錯位).
   - v17-E.50 contact `16.50 cm` < GT VQ roundtrip `18.47 cm` is a
     metric-gaming red flag.
2. **Architecture-level γ_int audit** (D-A) — should we focus on
   inference TTT or fix Stage B's IntXAttn conditioning?
3. **Inference-side route 1** — add Gumbel-Softmax / Concrete relaxation
   to per-step inner loop (the last unmatched MaskControl-equivalent
   piece I diff'd from `exitudio/ControlMM` source on 2026-05-01).

This doc captures D-A's result and the Gumbel-noise addition. v17-F
sweep on the server consumes these findings.

## D-A — γ_int (IntXAttn gate) audit

`train_generator.py:704-707` already logs `gamma_int_abs_mean` and
`gamma_int_res_abs_mean` to wandb (mean of |γ| across the 8
transformer layers). Reading the full v14 / v15 / v16 training
trajectories from
`runs/wandb_logs/wandb_history_genB_v{14,15,16}*.csv`:

| run | epoch 1 γ_int (base) | epoch 80 γ_int (base) | epoch 80 γ_int_res |
|---|---:|---:|---:|
| v14 sampled-ST | 5.5e-5 (zero-init) | **0.0176** | 0.0093 |
| v15 alignment-guided | 5.4e-5 | **0.0174** | 0.0098 |
| v16 alignment-mirror | 2.8e-4 | **0.0204** | 0.0113 |

Final `|γ_int|` ≈ **0.02** in all three runs. ControlNet-style
zero-init gates typically grow to **0.5–1.0** over comparable
training; PIANO is at **1/25 of typical**. IntXAttn is gated nearly
shut.

**However**: qual_eval's swap condition shows z_int matters
(full 26.79 cm vs swap 69.67 cm = 43 cm gap), so IntXAttn isn't doing
zero work. The 0.02 gate × 8 layers compounds into a non-trivial but
heavily-attenuated z_int signal path.

**Implication for next branches**:

- The **v9–v16 training-time decoded contact loss** has been doing
  most of its work via the **direct loss gradient** on the decoded
  motion, not via amplifying the IntXAttn cross-attention path. The
  loss adjusts what the base path generates rather than how strongly
  it attends to z_int.
- **v17 inference TTT** sidesteps the architectural gate entirely by
  optimising logits in decoded space. This explains why v17 produces
  10+ cm of contact gain in one inference change after 6 months of
  training-side iteration produced ~2 cm.
- **There is large unrealised lever in re-architecting the
  conditioning path** (e.g., remove γ zero-init; add a hard-bypass
  channel that injects contact_target_xyz directly into base logits;
  swap IntXAttn for ControlNet-style zero-init connectors with a
  warm-start). Defer until v17-F result; if v17-F pushes correct-part
  recall through 0.30, architectural rework may not be required.

## MaskControl uses pretrained MoMask VQ → codebook is not the bottleneck

User pushed back on the earlier "GT VQ roundtrip 18.47 cm could be a
codebook ceiling" framing, citing MaskControl's setup. Source-verified
2026-05-01 from `exitudio/ControlMM`:

| component | MaskControl train state | source |
|---|---|---|
| MoMask VQ-VAE (codebooks + decoder) | **frozen, pretrained** | `ctrl_train()` doesn't unfreeze vq_model |
| MoMask MaskTransformer base | **frozen** | `ctrl_train()` |
| Control encoder + parallel control transformer + zero-init connectors | **trained** | `ctrl_train()` |
| Training loss | **pure masked-CE on tokens** | no decoded geometric loss in training |

MaskControl achieves SOTA contact alignment with this setup → the
MoMask codebook **is sufficient** to express precise contact
configurations. The 18.47 cm GT VQ roundtrip number is "vanilla MoMask
training-objective reconstruction quality", not a codebook capacity
limit.

So: **deprioritise codebook re-training**. PIANO's bottleneck is in
the MaskTransformer + conditioning + training recipe, not the VQ.

## Route 1 — Gumbel-noise injection in per-step inner loop (v17-F)

### Source diff (PIANO vs MaskControl per-iter)

**MaskControl `each_iter` block** (`exitudio/ControlMM@models/mask_transformer/control_transformer.py::generate`):

```python
filtered_logits.requires_grad = True
optimizer = torch.optim.AdamW([filtered_logits], lr=lr, betas=(0.5, 0.9), weight_decay=1e-6)
for i in range(iter):
    _probs = ((filtered_logits / max(temperature, 1e-10)) + gumbel_noise(filtered_logits))
    emb = F.softmax(_probs, dim=-1) @ vq_model.quantizer.codebooks[0]
    ...
```

**PIANO v17-C/D/E** (`src/piano/inference/contact_guidance.py::_decode_with_relaxed_masked_base`):

```python
soft_emb = F.softmax(base_logits / max(temperature, 1e-6), dim=-1) @ base_codebook
```

The diff: MaskControl adds standard Gumbel(0, 1) noise to logits *before* softmax — Concrete /
Gumbel-Softmax relaxation. PIANO uses the noise-free softmax expectation. Effect:

- **MaskControl**: each inner step computes the relaxed embedding *under a sampled token
  perturbation*. Loss landscape ≈ expected loss over Gumbel samples → optimisation favours
  configurations that are good when actually sampled at commit time.
- **PIANO pre-v17-F**: each inner step computes the pure expectation. Loss landscape favours
  configurations with low expected distance, but the model is then committed via top-k +
  gumbel_sample in the actual MaskGIT step → train/test relaxation mismatch.

This is the largest unmatched diff in the source comparison and the
most directly principled fix. ~5 LOC code change.

### Implementation

`_decode_with_relaxed_masked_base` gains a new `gumbel_noise_scale: float = 0.0` parameter.
When > 0, applies `noisy_logits = logits + scale * (-log(-log(U)))` (standard Gumbel(0, 1))
before softmax. Wired up the chain:

- `_decode_with_relaxed_masked_base(...gumbel_noise_scale=...)`.
- `_generate_with_per_step_guidance(...per_step_gumbel_scale: float = 1.0)` (default ON to
  match MaskControl).
- `guide_with_contact(...per_step_gumbel_scale: float = 1.0)`.
- `qual_eval.py --per-step-gumbel-scale FLOAT` (default 1.0).
- `run_v13_target_trajectory.sh PER_STEP_GUMBEL_SCALE` env var (default 1.0).
- `guidance_trace.json` records `per_step_gumbel_scale` in both top-level config and
  per-clip `info.per_step` schema for verification on the next sync.

Inlined Gumbel noise (not imported from MoMask tools) so the helper is CPU-importable
without the backbone adapter. Numerically identical (same `-log(-log(U))` formula).

Backward compat: `gumbel_noise_scale=0.0` reproduces v17-C/D/E exactly. Tests:

- `test_decode_with_relaxed_masked_base_gumbel_noise_zero_matches_no_noise`
- `test_decode_with_relaxed_masked_base_gumbel_noise_changes_output`

Both pass.

### v17-F sweep plan (`scripts/stage_b_generator/run_v17f_gumbel_sweep.sh`)

Four conditions on the v16 best_contact ckpt:

| variant | per_step | gumbel | role |
|---|---:|---:|---|
| v17-F.10 | 10 | 1.0 | canonical MaskControl `each_iter`; ship candidate |
| v17-F.20 | 20 | 1.0 | does Gumbel + bigger budget compound? |
| v17-C-ng | 10 | 0.0 | sanity: must reproduce v17-C 21.77 cm contact |
| v17-E.20-ng | 20 | 0.0 | sanity: must reproduce v17-E.20 18.62 cm contact |

The two "ng" (no-Gumbel) conditions are reproducibility checks. If they
match v17-C / v17-E.20 exactly, we know the only difference vs the
sweep already on disk is the Gumbel addition.

Decision rule:

- **v17-F.10 ≥ v17-E.20 quality**: Gumbel beats raw budget; ship Gumbel + per_step=10
  (same wallclock as v17-C, v17-E.20-quality result). Update default `per_step_iters=10`
  in `run_v17_per_step_guidance.sh`.
- **v17-F.20 > v17-E.50**: Gumbel + budget compounds. Consider v17-F.50 next.
- **v17-F.10 ≈ v17-C**: Gumbel is neutral on this generator; the bottleneck is elsewhere
  (architectural; do γ_int rework branch from D-A).

Wallclock: 4 × ~70-90 min ≈ 5-6 h on a single A6000.

## Source quotes (verification provenance)

For future readers wanting to re-verify the MaskControl per-iter
recipe without re-fetching the source:

```python
# exitudio/ControlMM @ models/mask_transformer/control_transformer.py
# generate() per-iter block (each_iter > 0):
filtered_logits.requires_grad = True
optimizer = torch.optim.AdamW([filtered_logits], lr=lr, betas=(0.5, 0.9), weight_decay=1e-6)
if each_iter > 0:
    iter = each_iter
else:
    iter = (steps+1)*(-each_iter)
for i in range(iter):
    _probs = ((filtered_logits / max(temperature, 1e-10)) + gumbel_noise(filtered_logits))
    emb = F.softmax(_probs, dim=-1) @ vq_model.quantizer.codebooks[0]
    emb = emb.masked_fill(padding_mask.unsqueeze(-1), 0.)
    pred_motions, pred_motions_denorm = self.forward_predmotion(emb)
    ...
    optimizer.zero_grad(); loss_tta.backward(); optimizer.step()
```

Note: MaskControl decodes with **base only** (no residual context); PIANO uses **frozen
baseline residual emb sum**. This is a separate diff (#1 in the prior writeup) deferred
until v17-F result clarifies whether per-step / Gumbel is enough.

## References

- Pinyoanuntapong, E. et al. *MaskControl: Spatially-Conditioned Generation of Discrete
  Motion via Logit Optimization*. **ICCV 2025**. arXiv:2410.10780. Source verified
  2026-05-01 at `https://raw.githubusercontent.com/exitudio/ControlMM/main/models/mask_transformer/control_transformer.py`.
- Internal: `analyses/2026-05-01_per_step_guidance_design.md` (per-step design baseline).
- Internal: `analyses/2026-05-01_v17_per_step_result.md` (v17-C/D/E.20/E.50 result).
- γ_int trajectories: `runs/wandb_logs/wandb_history_genB_v{14,15,16}*.csv`.
