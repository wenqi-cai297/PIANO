# R41 Code Review Fix Instructions for Claude Code

Date: 2026-06-02
Reviewer: Codex
Reviewed range: `6b32362..HEAD`, current HEAD observed as
`44c3a2f Split R41 cascade calibration out of the launcher`.

This document is for Claude Code. It records the remaining issues from
Codex's R41 code review and gives concrete fix instructions.

Important correction from the user:

- Do **not** treat the P0 `bs=64` OOM as a training blocker. P0 was
  single-GPU by design; R41 training runs on two GPUs (`CUDA_VISIBLE_DEVICES=0,2`).
  Do not change R41 training batch size just because of that P0 result.
  Only adjust `batch_size` / `gradient_accumulation_steps` if a real
  two-GPU training smoke/check9 proves it is needed.

The remaining blockers are not about memory. They are about experiment
correctness and result interpretability.

---

## 0. Review Discipline

Work in stages. After each stage:

1. Run the verification listed for that stage.
2. Do a local code review of your own diff.
3. Only then continue to the next stage.

Do not bundle all fixes and inspect only at the end.

At the end:

1. Run all global verification commands in section 6.
2. Commit and push.
3. Write the required return document:

   `analyses/2026-06-02_r41_return_for_codex.md`

The return document must include files changed, exact behavior changes,
verification output, known residual risks, and server commands.

---

## 1. Blocker: Calibration Is Still Overwritten by Launcher Regen

### Problem

The current intended flow is:

```bash
python scripts/stage_a_generator/round41_cascade_calibration.py
python scripts/stage_a_generator/round41_apply_calibration.py \
  --calibration analyses/round41_cascade_calibration/<stamp>.json --apply
bash scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh
```

But the launcher currently regenerates R41 configs before training:

- `scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh:103`
  logs `Regenerating R41 configs...`
- `scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh:104`
  runs `round41_make_stage1_cascade_configs.py`
- `scripts/stage_a_generator/round41_make_stage1_cascade_configs.py:99`
  sets `cascade.w_total = 1.0`

Therefore, any values written by:

- `scripts/stage_a_generator/round41_apply_calibration.py:91`

can be reset to `1.0` immediately before training. That makes calibration
effectively non-persistent.

### Required Fix

Change the launcher so config regeneration is **not** the default.

Recommended behavior:

- Add `ROUND41_REGEN_CONFIGS=0` env default.
- Add CLI flag `--regen-configs`.
- If `--regen-configs` is not set:
  - generate configs only when one or more expected R41 YAML files are
    missing;
  - otherwise leave existing configs untouched.
- If `--regen-configs` is set:
  - regenerate configs intentionally;
  - log loudly that existing calibration values may be overwritten.

Also add a pre-train config audit:

- For every selected variant, read and log:
  - `cascade.enabled`
  - `cascade.w_total`
  - `cascade.w_motion_mse`
  - `cascade.w_world_joint_vel`
  - `cascade.w_l_pos_full`
  - `cascade.w_anchor_joint_pos`
  - `cascade.pb1_checkpoint`
- For non-control cascade variants, warn if `w_total == 1.0` and no
  calibration report has been applied. Do not necessarily fail; warn
  clearly so the server operator sees it.

### Files

- `scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh`
- possibly `scripts/stage_a_generator/round41_make_stage1_cascade_configs.py`
  if you add CLI args in section 4.

### Verification

Run:

```bash
bash -n scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh
```

Then do this dry run:

```bash
export DATASETS_ROOT=/dummy
bash scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh --dry-run
```

The dry run must **not** regenerate configs unless configs are missing
or `--regen-configs` is passed.

Also verify manually:

1. Apply a temporary `cascade.w_total` value to one generated config.
2. Run the launcher dry-run without `--regen-configs`.
3. Confirm the value is preserved.

---

## 2. Blocker: Calibration Uses Loss Ratio, Not Gradient Ratio

### Problem

`round41_cascade_calibration.py` describes itself as a grad-scale probe,
but currently it only parses the smoke-test loss ratio:

- `scripts/stage_a_generator/round41_cascade_calibration.py:67`
  parses `R41 cascade weighted=... weighted/mse_x0=...`
- `scripts/stage_a_generator/round41_cascade_calibration.py:112`
  recommends `w_total` from that measured loss ratio
- `src/piano/training/train_stage1.py:1482`
  prints that loss ratio during `--smoke-test`

This is not equivalent to the gradient ratio reaching Stage-1. Frozen
PB1 adds a nontrivial Jacobian between Stage-1's 23-D output and PB1's
motion loss, so loss scale can be a bad proxy for optimization scale.

### Required Fix

Calibration should recommend `cascade.w_total` from the actual gradient
ratio:

```text
grad_norm(actual_weighted_cascade_loss -> Stage1 params)
/
grad_norm(Stage1 self loss -> Stage1 params)
```

Use the actual enabled cascade stack from the target config, not only
motion MSE. For A3/A4 this means world velocity, L_pos, and anchor terms
must be included with the config's weights.

### Recommended Implementation

Prefer reusing/extending the existing P0 diagnostic path, because it
already has explicit gradient checks:

- `scripts/stage_a_generator/round41_stage1_cascade_p0_diag.py`
- `check_6_grad_scale(...)`

However, current P0 check 6 only measures motion-MSE cascade at `w=1`.
Extend it rather than blindly reusing it.

Concrete plan:

1. In `round41_stage1_cascade_p0_diag.py`, add a fast calibration mode,
   for example:

   ```bash
   --calibration-only
   ```

   It should run only the checks needed for calibration:

   - batch contract
   - Stage-1 warm-start load
   - PB1 ckpt/freeze load
   - cascade grad path
   - actual cascade grad scale

   Skip memory timing and distribution alignment in this mode.

2. Add a new check, for example `grad_scale_actual_stack`.

   It should:

   - read the target Stage-1 config's `cascade` block;
   - compute Stage-1 self loss gradient norm;
   - compute actual cascade loss:

     ```python
     cascade_loss_raw = (
         w_motion_mse * motion_mse
         + w_world_joint_vel * world_vel
         + w_l_pos_full * l_pos_full
         + w_anchor_joint_pos * anchor_pos
     )
     cascade_loss_weighted = w_total * cascade_loss_raw
     ```

   - backprop `cascade_loss_weighted`;
   - report:

     ```json
     {
       "grad_norm_stage1_self": ...,
       "grad_norm_actual_cascade_weighted": ...,
       "ratio_actual_cascade_over_self": ...
     }
     ```

3. To compute the actual stack, import and use the existing helper
   functions from:

   - `src/piano/training/pb1_loss_helpers.py`

   Do not reimplement FK, velocity loss, anchor loss, or min-SNR logic
   from scratch.

4. Update `round41_cascade_calibration.py` to call the fast P0
   calibration mode per config and parse the JSON value:

   ```text
   checks.grad_scale_actual_stack.ratio_actual_cascade_over_self
   ```

5. Recommend:

   ```python
   new_w_total = current_w_total * target_center / measured_grad_ratio
   ```

   Keep the target band default `[0.5, 1.5]`.

6. Keep the old loss ratio in the report as informational only. Do not
   use it for `w_total` recommendation.

### Files

- `scripts/stage_a_generator/round41_stage1_cascade_p0_diag.py`
- `scripts/stage_a_generator/round41_cascade_calibration.py`
- possibly tests under `tests/`

### Verification

Run syntax checks:

```bash
python -m py_compile \
  scripts/stage_a_generator/round41_stage1_cascade_p0_diag.py \
  scripts/stage_a_generator/round41_cascade_calibration.py
```

On the server, run:

```bash
python -u scripts/stage_a_generator/round41_cascade_calibration.py \
  --cfgs configs/training/stage1_r41_a1_motion_mse.yaml \
  --out-dir analyses/round41_cascade_calibration_smoke
```

The resulting JSON/MD must include both:

- actual grad ratio used for recommendation;
- loss ratio marked as informational.

---

## 3. Missing: Automatic Diagnostics Are Incomplete

### Problem

The launcher currently runs only direct Stage-1 -> PB1 downstream diag:

- `scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh:266`
  starts direct diag

The packer currently includes direct diag and P0 if present:

- `scripts/stage_a_generator/pack_round41_cascade_sync.sh:59`
  packs `analyses/round41_stage1_direct_diag_${VID}`
- `scripts/stage_a_generator/pack_round41_cascade_sync.sh:68`
  packs `analyses/round41_p0_cascade_diag`

This is not enough to judge the R41 hypothesis. Direct drift tells us
whether frozen PB1 likes the new Stage-1 output, but not whether Stage-1
collapsed or lost diversity.

### Required Fix

Extend the launcher and packer so post-train diagnostics include:

1. Direct Stage-1 -> PB1 downstream diag
   - already present; keep it.
2. R35-style Stage-1 coarse OOD audit
   - required by default.
3. K-sample diversity audit
   - required by default.
4. Full cascade diag
   - optional risk monitor, controlled by env/flag.

### Concrete Calls

For R35-style audit, use:

- `scripts/stage_a_generator/round35_stage1_coarse_ood_audit.py`

It expects:

```bash
--config
--generated-dir
--selection-json
--bucket
--out-md
--out-json
```

The direct diag script creates generated Stage-1 substitute cond caches
under a `round31_stage1_substitute_conds...` directory before moving the
diag output. Preserve or archive the generated-dir path so R35 can read
it. Do not delete it before R35/K-diversity if those diagnostics need it.

For K-diversity, use:

- `scripts/stage_a_generator/round40_stage1_k_sample_audit.py`

It expects:

```bash
--config
--ckpt
--selection-json
--bucket
--out-dir
--num-samples
--cfg-scale
```

Suggested defaults:

- `ROUND41_KDIV_NUM_SAMPLES=8`
- `ROUND41_KDIV_CFG_SCALE=1.0`

For full cascade diag, use:

- `scripts/stage_a_generator/run_round32_stage1p5_downstream_diag.sh`

This should stay optional because R41 does not retrain Stage-1.5. Add:

- `ROUND41_WITH_FULL_CASCADE=0` default
- CLI flag `--with-full-cascade`

If enabled, set:

```bash
ROUND32_DS_UPSTREAM_DIR=<R41 generated Stage-1 substitute cond dir>
ROUND32_DS_OUT_TAG="_r41_${VID}"
ROUND32_DS_BUCKETS="${BUCKETS_STR}"
```

Use the current best Stage-1.5 config/ckpt unless env overrides are set:

```bash
ROUND41_STAGE1P5_CFG=...
ROUND41_STAGE1P5_CKPT=...
```

### Packer Updates

Update `pack_round41_cascade_sync.sh` to include:

- calibration reports:
  - `analyses/round41_cascade_calibration`
- direct diag:
  - already included
- generated Stage-1 substitute cond directory metadata or summaries
  - exclude large `.npz` by default unless `ROUND41_PACK_NPZ=1`
- R35 audit dirs:
  - e.g. `analyses/round41_stage1_ood_${VID}`
- K-diversity dirs:
  - e.g. `analyses/round41_stage1_kdiv_${VID}`
- full cascade summaries if run:
  - e.g. `analyses/round41_full_cascade_${VID}` or the chosen Round32
    output dir
- matrix summary markdown/log
- return document:
  - `analyses/2026-06-02_r41_return_for_codex.md`

### Verification

Run:

```bash
bash -n scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh \
  scripts/stage_a_generator/pack_round41_cascade_sync.sh
```

Then dry-run:

```bash
export DATASETS_ROOT=/dummy
bash scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh \
  --dry-run --only stage1_r41_a1_motion_mse
```

Dry-run output must show the planned direct diag, R35 audit, K-diversity
audit, and whether full cascade is disabled/enabled.

---

## 4. Bug: PB1 Checkpoint Override Can Diverge Between Train and Diag

### Problem

The launcher supports:

- `ROUND41_PB1_CKPT`

and direct diag receives:

- `ROUND31_DS_PB1_CKPT="${PB1_CKPT}"`

But Stage-1 training loads PB1 from the YAML:

- `src/piano/training/train_stage1.py:1285`
  reads `cascade.pb1_checkpoint`

The generated YAML currently hardcodes:

- `scripts/stage_a_generator/round41_make_stage1_cascade_configs.py:98`
  `DEFAULT_PB1_CKPT`

So if the user overrides `ROUND41_PB1_CKPT`, training can use one PB1
while direct diag uses another. That invalidates the experiment.

### Required Fix

Make PB1 config/ckpt a single source of truth.

Recommended approach:

1. Add generator CLI args:

   ```bash
   --pb1-config
   --pb1-ckpt
   --init-checkpoint
   ```

2. In the launcher, when config generation is needed or requested, pass:

   ```bash
   --pb1-config "${PB1_CFG}"
   --pb1-ckpt "${PB1_CKPT}"
   ```

3. During launcher preflight, read each selected YAML and verify:

   ```text
   cfg.cascade.pb1_checkpoint == PB1_CKPT
   ```

   If not equal, fail with a clear message unless the user passed an
   explicit override like `ROUND41_ALLOW_PB1_CKPT_MISMATCH=1`.

4. Calibration should read PB1 paths from each YAML, not from a separate
   environment variable. This ensures it probes exactly what training
   will use.

### Verification

Dry-run two cases:

```bash
export ROUND41_PB1_CKPT=runs/training/stageB_anchordiff_r29_pb_a1_adaln_s4/final.pt
bash scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh --dry-run
```

Then temporarily set a different `ROUND41_PB1_CKPT`; the launcher should
either regenerate configs with that path or fail before training because
YAML and env disagree.

---

## 5. Missing Return Document

### Problem

The handoff required:

```text
analyses/2026-06-02_r41_return_for_codex.md
```

It does not currently exist.

### Required Fix

After implementing all fixes, write that file. It must include:

1. Files changed.
2. Exact implementation choices.
3. Calibration behavior:
   - whether it uses grad ratio;
   - how `w_total` is applied;
   - how launcher avoids overwriting it.
4. Diagnostics now run automatically.
5. Verification commands and outputs.
6. Server workflow commands.
7. Known residual risks.
8. Commit SHA after push.

Update the packer to include this document if present.

---

## 6. Global Verification Before Commit

Run locally:

```bash
python -m py_compile \
  src/piano/training/train_stage1.py \
  src/piano/training/pb1_loss_helpers.py \
  scripts/stage_a_generator/round41_make_stage1_cascade_configs.py \
  scripts/stage_a_generator/round41_stage1_cascade_p0_diag.py \
  scripts/stage_a_generator/round41_cascade_calibration.py \
  scripts/stage_a_generator/round41_apply_calibration.py

bash -n \
  scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh \
  scripts/stage_a_generator/pack_round41_cascade_sync.sh
```

If the environment has pytest installed:

```bash
python -m pytest -q \
  tests/test_stage1_init_checkpoint.py \
  tests/test_pb1_loss_helpers.py
```

On the server, run at least:

```bash
git pull --ff-only origin master
conda activate piano

export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
export ROUND41_GPUS="0,2"
export ROUND41_NUM_PROCESSES=2
export ROUND41_BUCKETS="val"

# Generate only if configs are missing, or explicitly:
bash scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh --dry-run

# Calibration phase:
python -u scripts/stage_a_generator/round41_cascade_calibration.py \
  --out-dir analyses/round41_cascade_calibration

python scripts/stage_a_generator/round41_apply_calibration.py \
  --calibration analyses/round41_cascade_calibration/<stamp>.json --apply

# Re-run calibration and confirm all cascade cells are in band:
python -u scripts/stage_a_generator/round41_cascade_calibration.py \
  --out-dir analyses/round41_cascade_calibration

# Launch matrix:
bash scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh
```

After completion, confirm the tarball includes the new reports:

```bash
tar -tzf analyses/round41_cascade_results_*.tar.gz | grep -E \
  'round41_cascade_calibration|round41_stage1_direct_diag|round41_stage1_ood|round41_stage1_kdiv|2026-06-02_r41_return_for_codex'
```

---

## 7. Final Commit / Push

After all fixes and verification:

```bash
git status --short
git add \
  scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh \
  scripts/stage_a_generator/pack_round41_cascade_sync.sh \
  scripts/stage_a_generator/round41_make_stage1_cascade_configs.py \
  scripts/stage_a_generator/round41_stage1_cascade_p0_diag.py \
  scripts/stage_a_generator/round41_cascade_calibration.py \
  scripts/stage_a_generator/round41_apply_calibration.py \
  analyses/2026-06-02_r41_return_for_codex.md

git commit -m "Fix R41 calibration persistence and diagnostics"
git push
```

Do not add unrelated untracked generated config files unless they are
intentionally part of this fix.
