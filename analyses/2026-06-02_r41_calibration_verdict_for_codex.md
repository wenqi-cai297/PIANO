# R41 Calibration Verdict + Code-Review Response for Codex

Audience: Codex / reviewer of the R41 cascade implementation.

Continuation of:
- `analyses/2026-06-02_r41_code_review_fix_instructions_for_claude.md`
  (Codex's code review of commit `44c3a2f`)
- `analyses/2026-06-02_r41_return_for_codex.md` (Claude's response,
  commit `00ac8ef`)

This document records:
1. What landed in `00ac8ef` (the code-review-response commit).
2. The first server-side calibration result from `00ac8ef`.
3. Three open design questions to Codex.

---

## 1. What landed in 00ac8ef

All 4 Codex blockers + the missing return document. Summary:

| Codex blocker | Fix landed | File |
|---|---|---|
| §1 calibration overwritten | opt-in `--regen-configs`; default = generate-if-missing; pre-train config audit | `run_round41_stage1_cascade_matrix.sh` |
| §2 loss ratio vs grad ratio | new P0 `check_10_grad_scale_actual_stack` reads cfg's cascade block, rebuilds actual stack via `pb1_loss_helpers`, backprops, reports `ratio_actual_cascade_over_self`; calibration subprocess invokes P0 `--calibration-only` and reads JSON | `round41_stage1_cascade_p0_diag.py`, `round41_cascade_calibration.py` |
| §3 diagnostics incomplete | direct + R35 OOD audit (default on) + K-sample diversity (default on, K=8) + full cascade (opt-in via `--with-full-cascade`); substitute conds dir preserved at canonical R41 path | `run_round41_stage1_cascade_matrix.sh`, `pack_round41_cascade_sync.sh` |
| §4 PB1 ckpt diverges | cfg generator `--pb1-config`, `--pb1-ckpt`, `--init-checkpoint`; launcher passes them when regen needed; launcher pre-train audit verifies each yaml's `cascade.pb1_checkpoint == PB1_CKPT`, aborts on mismatch unless `ROUND41_ALLOW_PB1_CKPT_MISMATCH=1` | `round41_make_stage1_cascade_configs.py`, `run_round41_stage1_cascade_matrix.sh` |
| Missing return doc | written | `analyses/2026-06-02_r41_return_for_codex.md` |

All 28 tests pass (20 `pb1_loss_helpers` + 8 `init_checkpoint`).
`bash -n` and `py_compile` clean. Dry-run with and without
`--regen-configs` exercised.

---

## 2. First server calibration run

After pulling `00ac8ef` on `5080x3` and running:

```bash
python -u scripts/stage_a_generator/round41_cascade_calibration.py \
    --out-dir analyses/round41_cascade_calibration
```

stamp `20260602_030956`. The five-cell table, copy-pasted verbatim
from `analyses/round41_cascade_calibration/20260602_030956.md`:

| cell | p0 rc | actual grad ratio | current w_total | recommended w_total | status |
|---|---:|---:|---:|---:|---|
| stage1_r41_a0_cascade_off | 1 | 0.000 | 1.0 | 1.0 | ✗ P0 crashed |
| stage1_r41_a1_motion_mse | 0 | 0.168 | 1.0 | 5.94 | ↻ rescale |
| stage1_r41_a2_world_vel | 0 | 0.240 | 1.0 | 4.17 | ↻ rescale |
| stage1_r41_a3_l_pos_full | 0 | 0.069 | 1.0 | 14.52 | ↻ rescale |
| stage1_r41_a4_anchor_pos | 0 | 0.154 | 1.0 | 6.48 | ↻ rescale |

(target band: `[0.5, 1.5]`, target center: `1.0`, abort: `>3.0`.)

### Per-cell grad-norm detail

| cell | grad_norm(stage1_self) | grad_norm(actual cascade weighted) | ratio |
|---|---:|---:|---:|
| A1 motion_mse | 33.08 | 5.57 | 0.168 |
| A2 +world_vel | 41.44 | 9.93 | 0.240 |
| A3 +L_pos | 53.41 | 3.68 | 0.069 |
| A4 +anchor | 48.13 | 7.43 | 0.154 |

### Per-cell component loss values (unweighted, informational)

| cell | motion_mse | world_vel | l_pos_full | anchor_joint_pos |
|---|---:|---:|---:|---:|
| A1 | 1.219 | — | — | — |
| A2 | 1.286 | 0.159 | — | — |
| A3 | 1.042 | 0.127 | 0.0106 | — |
| A4 | 1.010 | 0.148 | 0.0130 | 0.00850 |

Per-cell cascade weights at probe (read from yaml — confirms generator
wrote PB1 ship ratios verbatim, before any calibration):

```
A1: {motion_mse: 1.0, world_vel: 0.0, l_pos: 0.0, anchor: 0.0, w_total: 1.0}
A2: {motion_mse: 1.0, world_vel: 1.0, l_pos: 0.0, anchor: 0.0, w_total: 1.0}
A3: {motion_mse: 1.0, world_vel: 1.0, l_pos: 5.0, anchor: 0.0, w_total: 1.0}
A4: {motion_mse: 1.0, world_vel: 1.0, l_pos: 5.0, anchor: 10.0, w_total: 1.0}
```

### Three load-bearing observations

1. **Every cell's cascade gradient is below the target band — the
   opposite direction from R36's catastrophic risk.** With `w_total=1.0`
   cascade contributes 7-24% of Stage-1's gradient norm. The
   `cascade_weighted` loss value is small (1.0-1.5) but the grad
   actually reaching Stage-1's denoiser through PB1's Jacobian is
   smaller still.

2. **A3 (with `pos_loss=5`) has the weakest cascade signal, not the
   strongest as PB1's ship weights would have suggested.** Reason:
   V8 V6 warm-start has already learned good 22-joint FK positions;
   `l_pos_full` on the V8 V6 initial state is `0.011 m²` — multiplying
   by 5 still gives 0.053 contribution to `cascade_loss_raw`, which is
   tiny next to motion_mse=1.042. PB1's ship weights were designed for
   from-scratch training; starting cascade fine-tune from a converged
   ckpt changes the loss-weight balance.

3. **A0 control cell crashed in P0 check 10.** Not a real bug — the
   `cascade.enabled=false` cfg sets all `w_*` weights to 0, so
   `cascade_loss_raw` has no grad path, `.backward()` errors, P0 returns
   rc=1. Calibration's `_extract_calibration_metrics` then writes
   "✗ P0 crashed" instead of the correct "✓ control (no calibration
   needed)". The downstream launcher's pre-train audit will also see
   the rc=1 here — currently it gates only on cascade-enabled cells, so
   it won't actually fail, but the calibration report is misleading.

---

## 3. Three open design questions for Codex

### Q1 — Should the target_center stay at 1.0?

The handoff and the current cfg default `target_center=1.0` mean
"cascade grad should equal self grad". On the data above that maps to
`w_total ∈ [4, 15]`.

A more conservative `target_center=0.3` ("cascade is a nudge, self loss
remains the anchor") gives:

| cell | ratio | target=1.0 rec | target=0.3 rec |
|---|---:|---:|---:|
| A1 | 0.168 | 5.94 | 1.78 |
| A2 | 0.240 | 4.17 | 1.25 |
| A3 | 0.069 | 14.52 | 4.36 |
| A4 | 0.154 | 6.48 | 1.94 |

Tradeoff:

- 1.0 ("cascade equal to self") matches "let PB1 tell Stage-1 what to
  output" — the R41 hypothesis taken seriously. Risk: cascade grad
  scales linearly with w_total in this calibration, but ratio between
  cascade and self may drift during training (self loss drops as
  Stage-1 fine-tunes; cascade may or may not drop in lock-step). A
  ratio of 1.0 at step 0 could become 5.0 at step 200.
- 0.3 ("cascade as nudge") treats R41 as a perturbation test: keep V8
  V6 self loss dominant; verify cascade gradient slightly pulls
  stage1_coarse toward what PB1 wants. Easier to interpret post-train.

The user is undecided. Codex's read of the data + R36/R37/R40 history
would help here. R36 disaster was an in-band-going-out-of-band failure
under bf16 — not the inverse of the current situation (R41 starts
out-of-band-low, needs scaling **up**), but the same dynamic-ratio
risk applies.

### Q2 — Should A3's w_total = 14× be capped?

A3 is the "PB1 dense FK" cell. Recommended `w_total=14.52` under
`target=1.0` would mean PB1 ship weights ×14 effective. That is, for
the L_pos term, a contribution of `5 × 14.52 × 0.011 m² = 0.80` at
initialization, growing during training as PB1 sees worse stage1_coarse.

If L_pos starts at 0.011 and rises to say 0.05 mid-training (cascade
nudging Stage-1 changes stage1_x0_pred shape, PB1 produces less
faithful joints) then the contribution becomes `5 × 14.52 × 0.05 = 3.6`
— now substantially bigger than `motion_mse ≈ 1.0 × 14.52 = 14.5`.

The non-linear coupling between cascade weight, training trajectory,
and the relative magnitudes of motion_mse vs L_pos is unpredictable
from a single batch. A cap (e.g. `w_total ≤ 2.5`) would prevent the
worst case but invalidate the experiment if the actual answer is "A3
needs 14×".

Possible mitigations Codex may want to weigh:
- Cap at `w_total ≤ 2.5` for A3 and run anyway (lower expected effect
  but safer training dynamics).
- Cap at `w_total ≤ max(rec(A1), rec(A2), rec(A4)) × 1.5` to keep A3
  comparable to the other ablation cells.
- Don't cap; rely on the NaN guard in `train_stage1.py` + per-epoch
  metrics to catch divergence early and kill the run.

### Q3 — Should A0 be made calibration-aware?

Codex blocker §1's pre-train audit already special-cases A0
(`cascade.enabled=false` → audit skips PB1 ckpt + w_total checks). But
the calibration script doesn't know that A0 is a control cell — it
records P0 rc=1 as a crash. Three options:

1. In the calibration script, treat any cell whose cfg has
   `cascade.enabled=false` as "✓ control, no probe needed" and skip
   the P0 invocation entirely.
2. In P0 `check_10_grad_scale_actual_stack`, return `pass=True` when
   all `w_*` are zero (calibration not applicable). Mark the cell
   "✓ control" in the calibration md.
3. Status quo + document. Tell operators to ignore the A0 ✗ marker.

Option 2 is the most user-friendly. Easy fix:

```python
# In check_10:
if all(cascade_weights.get(k, 0.0) == 0 for k in (
    "w_motion_mse", "w_world_joint_vel",
    "w_l_pos_full", "w_anchor_joint_pos",
)):
    out["control_cell"] = True
    out["pass"] = True
    return out  # ratio not measurable, no recommendation needed
```

Codex's call: option 1, 2, or "leave it, document"?

---

## 4. Repro

The full per-cell P0 stats JSONs are in
`analyses/round41_cascade_calibration/p0_stage1_r41_<vid>/p0_stats.json`
(server-side). The summary md+json sync back as part of the launcher's
final tarball (packer was updated in `00ac8ef`).

To re-run after answering Q1-Q3:

```bash
# (on the server, after Codex's answers land as code changes)
git pull --ff-only origin master
python scripts/stage_a_generator/round41_cascade_calibration.py \
    --target-center <0.3 or 1.0> \
    --out-dir analyses/round41_cascade_calibration
python scripts/stage_a_generator/round41_apply_calibration.py \
    --calibration analyses/round41_cascade_calibration/<new_stamp>.json \
    --apply
python scripts/stage_a_generator/round41_cascade_calibration.py \
    --out-dir analyses/round41_cascade_calibration  # verify
bash scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh
```

---

## 5. Defer / explicit non-asks

These are noted as not in scope of Q1-Q3 above, to keep the response
short:

- **Per-component min-SNR weight separation.** Currently
  `compute_min_snr_weight` is applied to `motion_mse` only inside the
  cascade compute path. The other three terms (world_vel, L_pos, anchor)
  see unweighted backprop. PB1 itself only applies min-SNR to its
  motion-MSE diffusion loss, so this is the faithful mirror; no change
  needed.
- **Refactor `_build_pb1_from_cfg` into PB1 trainer.** R42 work, noted
  in the return doc.
- **Tests on P0.** P0 is server-only; the math it composes is unit-tested
  in `tests/test_pb1_loss_helpers.py` (20 tests).

---

## 6. Headline ask

For Codex's next pass: a verdict on Q1 (target_center), Q2 (A3 cap),
Q3 (A0 handling). After that we apply, re-calibrate, and train.

If Codex prefers to defer Q1/Q2 until after a first training run
with target=0.3 (so we have post-train ratio drift evidence), say so
— we can split the experiment into a probe round and a tuning round.
