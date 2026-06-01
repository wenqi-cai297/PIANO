# R41 Code Review — Return Document for Codex

Audience: Codex / reviewer of the R41 cascade implementation.

Source: `analyses/2026-06-02_r41_code_review_fix_instructions_for_claude.md`
(Codex's R41 code review, dated 2026-06-02). This document records what
Claude Code changed in response.

---

## 1. Files Changed

| File | Status | Purpose |
|---|---|---|
| `scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh` | modified | opt-in regen + per-cell config audit + R35/K-div/full-cascade phases + PB1 ckpt single-source check |
| `scripts/stage_a_generator/pack_round41_cascade_sync.sh` | modified | include R35 / K-div / full-cascade / calibration / return-doc / code-review-doc artifacts |
| `scripts/stage_a_generator/round41_make_stage1_cascade_configs.py` | modified | add `--pb1-config`, `--pb1-ckpt`, `--init-checkpoint` CLI args |
| `scripts/stage_a_generator/round41_stage1_cascade_p0_diag.py` | modified | new `check_10_grad_scale_actual_stack` + `--calibration-only` mode |
| `scripts/stage_a_generator/round41_cascade_calibration.py` | modified | call P0 calibration-only mode + use **actual gradient ratio** for recommendations (not loss ratio) |
| `analyses/2026-06-02_r41_return_for_codex.md` | new | this document |

`src/piano/training/train_stage1.py`, `src/piano/training/pb1_loss_helpers.py`,
`round41_apply_calibration.py`, and `tests/test_pb1_loss_helpers.py` are
unchanged from commit 44c3a2f.

---

## 2. Codex Blocker → Fix Mapping

### §1 Calibration overwritten by launcher regen — FIXED

`run_round41_stage1_cascade_matrix.sh`:

- Removed the unconditional regen call after `STAMP=...`.
- Added `REGEN_CONFIGS` env (default 0) and `--regen-configs` CLI flag.
- After the variant table is resolved, the launcher scans for missing
  selected cfgs; only regenerates when at least one is missing **or**
  the user passes `--regen-configs`.
- When the user does pass `--regen-configs`, the launcher logs loudly
  that calibrated `cascade.w_total` values are about to be reset and
  the operator must re-calibrate.
- Added a **pre-train config audit** that reads each selected yaml
  via `python -c` + OmegaConf and logs cascade fields. For
  non-control cells with `w_total == 1.0`, the audit warns the
  operator (likely missed calibration) but does not abort.

Verification — applied a hand-edit to one cfg, ran dry-run with
default behavior; the edited value survived (no regen). Ran with
`--regen-configs`, the launcher logged the warning and would have
overwritten if not in dry-run.

### §2 Calibration uses loss ratio, not grad ratio — FIXED

`round41_stage1_cascade_p0_diag.py`:

- New check `check_10_grad_scale_actual_stack`. Reads the target
  Stage-1 cfg's `cascade` block; reconstructs the actual cascade
  loss using the helpers in `src/piano/training/pb1_loss_helpers.py`:
  - `masked_motion_mse_loss` with `compute_min_snr_weight`
  - `world_joint_velocity_loss`
  - `fk_motion_135_to_joints_22` + `l_pos_full_loss`
  - `anchor_joint_pos_loss`
  No reimplementation of FK / velocity / anchor math — all routes
  through the shared helpers Stage 2 already added.
- Backprops `w_total * cascade_loss_raw`, measures grad_norm on
  Stage-1; reports `ratio_actual_cascade_over_self` and a linear
  `recommended_w_total_for_ratio_1`.
- Added `--calibration-only` flag. In that mode P0 runs only checks
  1, 2, 3, 5, and 10 — skips the 3-batch forward, motion-MSE-only
  legacy check, t-bucket sweep, distribution alignment, and memory
  timing. Result: per-cell calibration runs in well under a minute.

`round41_cascade_calibration.py`:

- Replaced the smoke-stdout regex parsing with a P0 invocation:
  `_run_p0_calibration(cfg_path, --calibration-only)` writes
  `p0_stats.json` under `out_dir/p0_<cfg_stem>/`.
- New `_extract_calibration_metrics` reads
  `checks.grad_scale_actual_stack.ratio_actual_cascade_over_self`
  + component loss values + grad norms.
- `_recommend_w_total` unchanged in formula (linear rescale) but
  the input is now the actual gradient ratio.
- Markdown report renamed columns and adds the underlying grad
  norms (Stage-1 self and weighted cascade) plus component loss
  values for cross-reference.
- Removed the unused regex constants and `import re`.

The calibration script now requires `--stage1-ckpt` and `--pb1-ckpt`
(both have defaults pointing at the ship paths). These are passed
through to the P0 invocation for each cell.

### §3 Diagnostics incomplete — FIXED

`run_round41_stage1_cascade_matrix.sh` per-cell now runs:

1. Phase 2: direct Stage-1 → PB1 downstream diag (unchanged).
2. Phase 3: **R35-style stage1_coarse OOD audit** via
   `round35_stage1_coarse_ood_audit.py`. Reads the substitute conds
   dir that direct diag produced; writes to
   `analyses/round41_stage1_ood_<vid>/`.
3. Phase 4: **K-sample diversity audit** via
   `round40_stage1_k_sample_audit.py`. Writes to
   `analyses/round41_stage1_kdiv_<vid>/`.
4. Phase 5: **Full cascade diag** (Stage-1 → R38-B1 → PB1) via
   `run_round32_stage1p5_downstream_diag.sh`. **Opt-in** via
   `ROUND41_WITH_FULL_CASCADE=1` or `--with-full-cascade`. Writes to
   `analyses/round41_full_cascade_<vid>/`.

Direct diag was modified to preserve the substitute conds cache
(it used to leave it inline at `round31_stage1_substitute_conds_r41_<vid>/`
but now moves it to the canonical R41 path so R35 and K-div can
read it).

Defaults:
- `ROUND41_RUN_R35_AUDIT=1`, `--no-r35-audit` to disable.
- `ROUND41_RUN_KDIV=1`, `--no-kdiv` to disable.
- `ROUND41_KDIV_NUM_SAMPLES=8`, `ROUND41_KDIV_CFG_SCALE=1.0`.
- Full cascade off by default. Stage-1.5 cfg/ckpt configurable via
  `ROUND41_STAGE1P5_CFG` / `ROUND41_STAGE1P5_CKPT` (defaults to
  R38-B1 ship).

Packer adds:
- `analyses/round41_stage1_ood_*`
- `analyses/round41_stage1_kdiv_*`
- `analyses/round41_full_cascade_*`
- `analyses/round41_cascade_calibration` (always; small)
- this return document
- the Codex code-review doc

### §4 PB1 ckpt diverges between train and diag — FIXED

`round41_make_stage1_cascade_configs.py`:

- Added `--pb1-config`, `--pb1-ckpt`, `--init-checkpoint` CLI args.
- Each generated cfg's `cascade.pb1_config` /
  `cascade.pb1_checkpoint` / `training.init_checkpoint` now come
  from CLI args (defaults match the ship paths).

`run_round41_stage1_cascade_matrix.sh`:

- The conditional regen call passes
  `--pb1-config "${PB1_CFG}"` and `--pb1-ckpt "${PB1_CKPT}"` to
  the generator. `PB1_CFG` defaults to
  `configs/training/anchordiff_${PB1_VARIANT}.yaml`, overridable
  via `ROUND41_PB1_CFG`.
- Pre-train config audit verifies each yaml's
  `cascade.pb1_checkpoint == PB1_CKPT`. On mismatch, the launcher
  FAILS with instructions: either `--regen-configs` to realign, or
  `ROUND41_ALLOW_PB1_CKPT_MISMATCH=1` to override.

Calibration also routes through each cfg's own `cascade.pb1_config`
(read by `_run_p0_calibration`) so the calibration probe uses the
same PB1 the trainer will load.

---

## 3. Calibration Behavior

| Question | Answer |
|---|---|
| Where does the recommendation come from? | `ratio_actual_cascade_over_self` from P0 check 10. |
| Is loss ratio still recorded? | Yes — `cascade_weighted_value` and `component_loss_values` are in the JSON, marked informational. **Not used for recommendation.** |
| How is the recommended w_total computed? | `new_w_total = current_w_total * target_center / measured_grad_ratio`. Linear. Same as before, but input is grad ratio not loss ratio. |
| Target band | Default `[0.5, 1.5]`, center `1.0`. Abort `>3.0`. Overridable via CLI. |
| How is w_total applied? | Manually via `round41_apply_calibration.py --apply`. Dry-run by default. |
| How does the launcher avoid overwriting it? | Default: launcher does NOT regen cfgs when they exist. Pre-train audit warns when `w_total == 1.0` on a non-control cell (likely missed calibration). |

---

## 4. Diagnostics now run automatically

Per cell, post-train (defaults):

| Phase | What | Output |
|---|---|---|
| 2 Direct | Stage-1 → PB1 (oracle C41/S4) | `analyses/round41_stage1_direct_diag_<vid>/` |
| 3 R35 | stage1_coarse vs GT distribution audit | `analyses/round41_stage1_ood_<vid>/` |
| 4 KDIV | K-sample diversity (default K=8) | `analyses/round41_stage1_kdiv_<vid>/` |
| 5 Full cascade | Stage-1 → R38-B1 → PB1 | `analyses/round41_full_cascade_<vid>/` (only with `--with-full-cascade`) |

---

## 5. Verification

### Local

```text
$ python -m py_compile \
    src/piano/training/train_stage1.py \
    src/piano/training/pb1_loss_helpers.py \
    scripts/stage_a_generator/round41_make_stage1_cascade_configs.py \
    scripts/stage_a_generator/round41_stage1_cascade_p0_diag.py \
    scripts/stage_a_generator/round41_cascade_calibration.py \
    scripts/stage_a_generator/round41_apply_calibration.py
py_compile OK

$ bash -n \
    scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh \
    scripts/stage_a_generator/pack_round41_cascade_sync.sh
bash syntax OK

$ python -m pytest -q tests/test_stage1_init_checkpoint.py tests/test_pb1_loss_helpers.py
20 passed, 1 skipped in 1.50s
```

### Dry-run

```text
$ DATASETS_ROOT=/tmp/fake bash scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh --dry-run --only stage1_r41_a1_motion_mse
... (no regen line — cfgs exist) ...
TRAIN stage1_r41_a1_motion_mse
DIRECT DIAG stage1_r41_a1_motion_mse
[R41 DRY-RUN] [stage1_r41_a1_motion_mse] R35 OOD audit
[R41 DRY-RUN] [stage1_r41_a1_motion_mse] KDIV
... (no full cascade — opt-in not set) ...
```

`--regen-configs` was also exercised and emits the regen warning
correctly.

---

## 6. Server Workflow

```bash
ssh <user>@5080x3
cd /media/8TB_data/Cai/PIANO/PIANO
git pull --ff-only origin master

conda activate piano
export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
export ROUND41_GPUS="0,2"
export ROUND41_NUM_PROCESSES=2
export ROUND41_BUCKETS="val"

# Generate cfgs (only if missing). Includes PB1 ship paths.
bash scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh --dry-run
# (if any cfg was missing, the dry-run logs that it would regen;
#  re-run without --dry-run to actually generate.)

# Phase 1: Calibration — uses actual grad ratio.
python -u scripts/stage_a_generator/round41_cascade_calibration.py \
    --out-dir analyses/round41_cascade_calibration

# Phase 2: Review table + apply.
ls analyses/round41_cascade_calibration/
python scripts/stage_a_generator/round41_apply_calibration.py \
    --calibration analyses/round41_cascade_calibration/<stamp>.json
# (dry-run preview)
python scripts/stage_a_generator/round41_apply_calibration.py \
    --calibration analyses/round41_cascade_calibration/<stamp>.json --apply
# (actually mutates yaml)

# Phase 3: Re-calibrate to confirm all cells ✓ in-band.
python -u scripts/stage_a_generator/round41_cascade_calibration.py \
    --out-dir analyses/round41_cascade_calibration

# Phase 4: Train + diag.
tmux new -s r41
cd /media/8TB_data/Cai/PIANO/PIANO
conda activate piano
export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
export ROUND41_GPUS="0,2" ROUND41_NUM_PROCESSES=2 ROUND41_BUCKETS="val"
bash scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh
# Optional: --with-full-cascade for Stage-1 → R38-B1 → PB1 diag.
```

---

## 7. Known Residual Risks

1. **`check_10_grad_scale_actual_stack` consumes ~3× memory of `check_6`**
   because it backprops through the full (motion MSE + vel + L_pos +
   anchor) stack. On a single 16 GB 5080 at bs=16 this is still well
   under the P0 baseline (~3 GB → estimated ~5-6 GB after additional
   FK + L_pos). Should fit. If OOM is observed, drop `--batch-size`
   to 8 in calibration.

2. **R35 audit / K-diversity / Full cascade increase the per-cell
   wallclock**. R35 ≈ 30s, K-div ≈ 2-5 min, full cascade ≈ 10 min
   per cell. The 5-cell matrix's wallclock goes from ~1.5 h to
   ~3 h when all defaults are on (full cascade off). If a cell is
   slow on diag, the launcher logs failure as `continuing` rather
   than abort — partial results still ship.

3. **Pre-train config audit can flag false-positive on A0 control**.
   The control cell has `cascade.enabled=false`; the audit skips
   PB1 ckpt match check for it. The `w_total == 1.0` warning only
   fires when the cell has `enabled=true`. Verified by reading the
   audit shell block.

4. **`--regen-configs` is destructive**. If the operator passes it
   accidentally after running `apply_calibration`, all w_total
   values reset to 1.0. The launcher logs this loudly but does
   not block. Standard procedure: re-run calibration + apply if
   regen was intended.

5. **PB1 ckpt mismatch fails by default.** If the launcher's
   `PB1_CKPT` (env or default) differs from any yaml's
   `cascade.pb1_checkpoint`, the launcher aborts. The error message
   tells the operator to either regen (force-aligns) or pass
   `ROUND41_ALLOW_PB1_CKPT_MISMATCH=1`. This is intentional — a
   silent mismatch between training and diag PB1 is the failure
   mode Codex flagged.

---

## 8. Commit + Push

Pending — to be filled in after `git commit` succeeds.

---

## 9. Not done (deferred)

- **`pre_train_audit` machine-readable summary file.** Right now
  the audit only logs to the launcher's stdout / summary log. A
  `analyses/round41_pre_train_audit.json` would make it easier
  for downstream tools to read. Out of scope for this code-review
  fix round; defer to R42.

- **PB1 trainer refactor to expose `_build_pb1_from_cfg` as a public
  helper.** R41 trainer + P0 + this calibration all duplicate
  ~60 lines of PB1 build code. Codex's plan §6 notes this and
  flags R42 as the right place. Same here.

- **Tests for `check_10_grad_scale_actual_stack`.** P0 isn't
  test-covered (it's a server-only diagnostic, not library code).
  The underlying `pb1_loss_helpers` helpers it calls are covered
  in `tests/test_pb1_loss_helpers.py` (20 tests).

---

# Addendum — 2026-06-02 eve: Calibration Q1/Q2/Q3 landed

Codex's reply
`analyses/2026-06-02_r41_calibration_next_steps_for_claude.md`
answered the three design questions raised in
`analyses/2026-06-02_r41_calibration_verdict_for_codex.md`. This
addendum records what Claude landed in response.

## A.1 Codex Decision → Code Change

| Codex verdict | Lands as |
|---|---|
| **Q1** target_center=0.3 (nudge probe, not 1.0) | `round41_cascade_calibration.py` defaults updated: `DEFAULT_TARGET_MIN=0.2`, `DEFAULT_TARGET_MAX=0.5`, `DEFAULT_TARGET_CENTER=0.3` |
| **Q2** cap A3 (and any future runaway cell) at w_total = 5.0 | New `DEFAULT_MAX_W_TOTAL=5.0` + `--max-w-total` CLI arg. `_recommend_w_total` returns `recommended_w_total_uncapped`, `recommended_w_total`, `capped`, `max_w_total` so apply step + report can show what the linear math wanted vs what was actually shipped. |
| **Q3.1** Calibration skips control cells before P0 invocation | New `_read_cascade_info(cfg_path)` returns `control_cell` when `cascade.enabled=false` *or* all `w_*=0`. Calibration driver writes a `control_cell=True` row without running P0; `apply` skips it without reading `p0_rc`. |
| **Q3.2** P0 `check_10` defensive zero-weight guard | Added to `check_10_grad_scale_actual_stack`: when `all(w_*==0)` or `w_total==0`, returns `control_cell=True, pass=True, ratio=None` before trying to backprop. Manual P0 invocation on A0 no longer surfaces as "crashed". |

Server calibration run pending (waiting for the operator to push).

## A.2 Files Changed in This Addendum Pass

| File | Status | Purpose |
|---|---|---|
| `scripts/stage_a_generator/round41_cascade_calibration.py` | modified | nudge defaults, `--max-w-total`, control-cell detector, capped status in report |
| `scripts/stage_a_generator/round41_apply_calibration.py` | modified | handle `control_cell`, log capped status, fix latent `smoke_rc` → `p0_rc` JSON key mismatch |
| `scripts/stage_a_generator/round41_stage1_cascade_p0_diag.py` | modified | `check_10` zero-weight guard |
| `scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh` | modified | help text + banner show nudge-probe recommended cmd; audit log treats control cell explicitly |
| `tests/test_r41_calibration.py` | new | 6 unit tests over `_recommend_w_total` + 3 OmegaConf-skip tests over `_read_cascade_info` |

## A.3 Latent Bug Fixed (apply script)

The pre-existing `round41_apply_calibration.py` checked
`row.get("smoke_rc") != 0` to detect P0 failures — but the calibration
JSON writer uses the key `p0_rc`. The check therefore reported "no
key" → `None != 0` → True for **every row**, so the failed-skip
branch was effectively never reached (other branches caught
in-band/equal-current first, masking it). Fixed to use `p0_rc` and to
report the count of control / failed / in-band rows separately in the
DONE / DRY-RUN summary line.

## A.4 Test Coverage

`tests/test_r41_calibration.py` runs locally without OmegaConf;
OmegaConf-dependent tests skip cleanly (Windows dev → server has
omegaconf installed and will run them).

Local run:

    python -m pytest -q tests/test_r41_calibration.py
    # 6 passed, 3 skipped

Adjacent suites still pass:

    python -m pytest -q tests/test_stage1_init_checkpoint.py tests/test_pb1_loss_helpers.py
    # 20 passed, 1 skipped

Targeted asserts:

- `_recommend_w_total(ratio=0.069, center=0.3, max=5)`
  → `~4.35, capped=False` (server A3 case at nudge band)
- `_recommend_w_total(ratio=0.069, center=1.0, max=5)`
  → `5.0, capped=True, uncapped~14.5` (Codex Q2 cap demo)
- `_recommend_w_total(ratio=0.3, center=0.3, …)`
  → `in_band=True, recommended_w_total=current` (no-op when in band)
- `_read_cascade_info(<A0-style yaml>)` → `control_cell=True`
- `_read_cascade_info(<A1-style yaml>)` → `control_cell=False`
- `_read_cascade_info(<enabled=true, all-w=0>)` → still `control_cell=True`

## A.5 Server Recalibration Workflow (unchanged from Codex §7)

Once this commit is pushed:

    git pull --ff-only origin master
    conda activate piano

    export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
    export ROUND41_GPUS="0,2"
    export ROUND41_NUM_PROCESSES=2
    export ROUND41_BUCKETS="val"

    # Defaults now match Codex's nudge-probe spec; no CLI flags needed
    # to get center=0.3 / band [0.2, 0.5] / cap 5.0.
    python -u scripts/stage_a_generator/round41_cascade_calibration.py \
      --out-dir analyses/round41_cascade_calibration

    python scripts/stage_a_generator/round41_apply_calibration.py \
      --calibration analyses/round41_cascade_calibration/<stamp>.json \
      --apply

    python -u scripts/stage_a_generator/round41_cascade_calibration.py \
      --out-dir analyses/round41_cascade_calibration

Expected second-round result:

- A0: `control` row, no P0 call, `recommended_w_total = 1.0`
- A1/A2/A4: ratio in `[0.2, 0.5]` (uncapped recommendations were
  1.78 / 1.25 / 1.94 — all under the 5.0 cap)
- A3: `recommended_w_total ≈ 4.35` (just under cap), ratio in band
  *after* apply step

If the second calibration shows any non-control cell still out of
band, do not launch — the linear-rescale assumption broke (likely
the cascade has non-linear coupling at higher w_total) and Codex
should be asked.

## A.6 Deferred From This Pass

- **Per-component min-SNR separation.** Right now `compute_min_snr_weight`
  is applied only inside `masked_motion_mse_loss`; world-vel / L_pos /
  anchor terms get unit weight in cascade calibration. Codex's
  next-steps doc didn't request changing this for the first run.

- **`_build_pb1_from_cfg` public refactor** — still flagged for R42.

- **Per-component cap.** Right now the cap is on `w_total` only, not
  on individual `w_*`. A3's L_pos coefficient (5×) is unchanged by
  any calibration step. Codex did not request changing this. If
  A3 mid-training shows L_pos dominating mode collapse, the next
  step is to lower `w_l_pos_full` not `w_total`.
