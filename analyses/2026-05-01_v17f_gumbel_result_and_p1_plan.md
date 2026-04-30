# 2026-05-01 — v17-F Gumbel sweep result (negative) + P1 γ_int boost plan

## v17-F result: Gumbel-Softmax injection is *negative* on PIANO

Per-step inner-loop Gumbel-noise addition (matching MaskControl's
canonical `each_iter` block; source-verified diff in
`analyses/2026-05-01_v17_diagnostics_and_gumbel.md`) was tested on the
v16 best_contact ckpt with the same per-step budgets as v17-C / v17-E.20:

| condition | contact | coupled | IoU | correct-part | local-err |
|---|---:|---:|---:|---:|---:|
| GT VQ roundtrip ceiling | 18.47 | – | – | – | – |
| v16 raw | 26.79 | 0.2734 | 0.382 | 0.176 | 53.49 |
| v17-C  (10, Gumbel OFF, original) | 21.77 | 0.3428 | 0.439 | 0.202 | 46.13 |
| **v17-C-ng** (10, Gumbel OFF, rerun sanity) | **21.80** | 0.3422 | 0.439 | 0.206 | 45.94 |
| **v17-F.10** (10, Gumbel ON) | **23.53** | 0.3251 | 0.403 | 0.177 | 49.02 |
| v17-E.20 (20, Gumbel OFF, original) | 18.62 | 0.3559 | 0.473 | 0.264 | 42.09 |
| **v17-E.20-ng** (20, Gumbel OFF, rerun sanity) | **18.19** | 0.3550 | 0.475 | 0.271 | 41.90 |
| **v17-F.20** (20, Gumbel ON) | **19.36** | 0.3196 | 0.472 | 0.219 | 42.79 |

Sanity rebuilds (Gumbel-OFF reruns) match originals within 0.5 cm —
pipeline is consistent. The Gumbel-ON regressions are real signal.

### Per-budget Gumbel deltas

**v17-F.10 vs v17-C-ng (per_step=10):**
- contact: +1.73 cm worse
- coupled: −0.017
- IoU: −0.036
- correct-part: −0.029
- local-err: +3.08 cm worse

**v17-F.20 vs v17-E.20-ng (per_step=20):**
- contact: +1.17 cm worse
- coupled: −0.035
- IoU: −0.003 (effectively tied)
- correct-part: −0.052 (largest single regression)
- local-err: +0.89 cm worse

**Gumbel-ON regresses every metric at both budgets.** Not neutral, not
mixed — uniformly negative.

### Why does MaskControl-canonical Gumbel work for them but hurt us

Most likely root cause: **multi-quantizer residual handling difference,
amplified by Gumbel noise**.

```
MaskControl per-iter (residual ignored):
  emb = softmax(logits + gumbel) @ codebook[0]
  motion = decoder(emb)                    ← base IS the only signal source

PIANO v17-F per-iter (frozen baseline residual sum):
  emb_base = softmax(logits + gumbel) @ codebook[0]
  x_emb = emb_base + baseline_residual_emb_sum
  motion = decoder(x_emb)                  ← residual_emb_sum dominates magnitude
```

In MaskControl's setting the base layer is the only motion signal so
Gumbel-Softmax is well-conditioned reparameterisation (each inner step
samples a different "what-if-this-token" trajectory). In PIANO the
frozen residual_emb_sum dominates the embedding magnitude, so the base
contribution is a small fraction. Adding Gumbel noise to that small
fraction makes the gradient signal noisy across inner steps — AdamW
(betas=(0.5, 0.9)) doesn't average it out.

This is consistent with the v17-D failure pattern (post-hoc full-RVQ
optimisation on top of per-step also regresses on PIANO): **the
multi-quantizer residual stack changes the curvature of the inner
optimization in ways the canonical MaskControl recipe doesn't account
for.**

### Decision

**Do NOT ship Gumbel.** Default `--per-step-gumbel-scale` revert to
`0.0` for the PIANO inference path. Keep the flag — it's the right
default for any single-quantizer reuse and ablation.

**v17-E.20 / v17-E.50 (Gumbel OFF) remain the ship configs:**

| ship | contact | coupled | IoU | correct-part | local | wallclock |
|---|---:|---:|---:|---:|---:|---|
| v17-E.20 | 18.62 | 0.3559 | 0.473 | 0.264 | 42.09 | ~70-90 min/80clips |
| v17-E.50 | 16.50 | 0.3533 | 0.504 | 0.275 | 39.02 | ~120-150 min/80clips |

v17-E.20 matches v14 K=64 alignment oracle on most metrics and is the
recommended default. v17-E.50 produces lower contact distance but
shows visible "metric gaming" warning signs (contact < GT roundtrip
ceiling, body-part patches still visibly misaligned in user review)
and should be paired with motion-quality QA before being shipped on
its own.

## P1 — inference-time γ_int boost (next branch)

The inference path has been near-saturated. v17-D (post-hoc stack), v17-F
(Gumbel) both negative; v17-E budget sweep shows diminishing returns.
Remaining inference levers are exhausted.

D-A audit (analyses/2026-05-01_v17_diagnostics_and_gumbel.md) showed
γ_int ≈ 0.02 in v14/v15/v16 — **IntXAttn cross-attention is gated to
~1/25 of typical ControlNet-style strength**, meaning the structured
z_int signal (contact_state, contact_target_xyz, phase, support) is
heavily attenuated as it flows into Stage B's base path. v17 inference
optimisation works partly *because* it sidesteps this gate, working
directly on logits in decoded space.

**P1 hypothesis**: the residual contact-patch misalignment user noted
in visual review is downstream of z_int being underused architecturally.
If we *boost γ_int at inference time* (multiply by a constant), the
generated motion should be more strongly conditioned on z_int from the
start, leaving less work for per-step to do.

### P1 design

Add `gamma_int_boost: float = 1.0` parameter through:

- `guide_with_contact(...gamma_int_boost=...)` — context-managed scaling
  of all `gamma_int` (and `gamma_int_res`) parameters in
  `transformer.mask_transformer.seqTransEncoder.layers` (and the
  residual transformer's analog) during the generation block. Restored
  on exit via `try/finally`.
- `qual_eval.py --gamma-int-boost FLOAT` (default 1.0).
- `run_v13_target_trajectory.sh GAMMA_INT_BOOST` env var (default 1.0).

The boost mutates the parameter `data` in-place under torch.no_grad()
for the duration of the inference call, then restores the original
values. This is cleaner than threading a multiplier through every
forward call — γ_int is consulted at every transformer layer at every
MaskGIT step, and a context manager touches one place.

### P1 ablation: v17-G boost sweep

Four conditions on v16 best_contact + v17-E.20 base config (per_step=20,
Gumbel OFF):

| variant | gamma_int_boost | role |
|---|---:|---|
| v17-G.b1 | 1.0 | sanity reproducer of v17-E.20 |
| v17-G.b2 | 2.0 | conservative — ~0.04 effective γ_int |
| v17-G.b5 | 5.0 | moderate — ~0.10 effective γ_int |
| v17-G.b10 | 10.0 | aggressive — ~0.20 effective γ_int (still 1/2.5 of typical) |
| v17-G.b20 | 20.0 | extreme — ~0.40 effective γ_int (approx ControlNet typical) |

Wallclock: 5 × ~80 min ≈ 7 h on a single A6000.

### Decision rule for P1

| outcome | implication | next |
|---|---|---|
| boost=2 already shows clear improvement (correct-part +3+ pp, local-err −3+ cm) | γ_int IS the bottleneck. Do P2 next: re-init γ_int with positive constant + finetune. | finalise the boost level + Stage B retrain plan |
| improvement plateaus at boost=5 / boost=10 | γ_int helps to a point then saturates (likely runs into joint distribution issue). Ship best boost as additional ship-config variant. | P2 retrain still useful but lower priority |
| boost=20 catastrophic (contact > 30 cm, OOD body) | the IntXAttn output magnitude × 0.4 is OOD for the trained base path. γ_int is a regulariser, not a capacity bottleneck. **Architectural rework needed (ControlNet-style hard-bypass channel, P3).** | P3 architecture branch becomes the long-term path |
| no monotonic trend | γ_int is not the lever; bottleneck is elsewhere. Pivot to OMOMO-style explicit contact_target as input (path 1 from earlier). | path 1 prototype |

## References

- Earlier v17 docs:
  - `analyses/2026-05-01_per_step_guidance_design.md` (design baseline).
  - `analyses/2026-05-01_v17_per_step_result.md` (v17-C + v17-D/E sweep result).
  - `analyses/2026-05-01_v17_diagnostics_and_gumbel.md` (γ_int audit + Gumbel implementation).
- Run artefacts:
  - `runs/eval/stageB_v0_17_v16bc_f10_gumbel_*` / `f20_gumbel_*` / `c_no_gumbel_*` / `e20_no_gumbel_*`.
- MaskControl source (verified 2026-05-01):
  `https://raw.githubusercontent.com/exitudio/ControlMM/main/models/mask_transformer/control_transformer.py`.
