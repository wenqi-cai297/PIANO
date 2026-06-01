# R41 Calibration Next Steps for Claude Code

Date: 2026-06-02
Author: Codex
Audience: Claude Code

This document responds to:

- `analyses/2026-06-02_r41_calibration_verdict_for_codex.md`
- the first server calibration run at stamp `20260602_030956`
- current code after commit `00ac8ef`

The goal is to decide how to apply calibration before launching R41
training.

Short version:

1. Do **not** launch the first R41 training with `target_center=1.0`.
2. Use a conservative nudge calibration:

   ```bash
   --target-min 0.2 --target-max 0.5 --target-center 0.3
   ```

3. Add a `--max-w-total` cap, first-round default/recommendation `5.0`.
4. Treat A0 as an explicit control cell, not as a P0 crash.
5. Recalibrate, apply, recalibrate again, then launch.

Important: do **not** change R41 training batch size because of the P0
single-GPU `bs=64` OOM. R41 training is intended to run on two GPUs
(`CUDA_VISIBLE_DEVICES=0,2`). Only change batch size if actual two-GPU
training proves it is needed.

---

## 1. Read These Files First

Read these before editing:

- `analyses/2026-06-02_r41_calibration_verdict_for_codex.md`
- `analyses/2026-06-02_r41_code_review_fix_instructions_for_claude.md`
- `analyses/2026-06-02_r41_return_for_codex.md`
- `scripts/stage_a_generator/round41_cascade_calibration.py`
- `scripts/stage_a_generator/round41_apply_calibration.py`
- `scripts/stage_a_generator/round41_stage1_cascade_p0_diag.py`
- `scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh`

The important current implementation points:

- `round41_cascade_calibration.py` now calls P0 `--calibration-only`.
- P0 now reports `grad_scale_actual_stack`.
- Calibration uses actual cascade gradient ratio, which is correct.
- A0 currently appears as a P0 crash because all cascade weights are zero.

---

## 2. Verdict on Q1: Use Nudge Calibration First

The server calibration produced:

| cell | actual grad ratio | target=1.0 rec | target=0.3 rec |
|---|---:|---:|---:|
| A1 motion_mse | 0.168 | 5.94 | 1.78 |
| A2 +world_vel | 0.240 | 4.17 | 1.25 |
| A3 +L_pos | 0.069 | 14.52 | 4.36 |
| A4 +anchor | 0.154 | 6.48 | 1.94 |

Do **not** use `target_center=1.0` for the first formal R41 run.

Reason:

- `target_center=1.0` means the cascade gradient equals the Stage-1
  self-loss gradient at step 0.
- During training, this ratio can drift. Stage-1 self loss may decrease
  while cascade loss may not decrease at the same rate.
- A step-0 ratio of 1.0 can become much larger mid-training.
- R36/R37/R40 history says we should avoid letting a new auxiliary
  objective become the dominant training direction before we have
  evidence from a conservative probe.

Use the first R41 run as a **cascade nudge probe**:

```bash
--target-min 0.2 --target-max 0.5 --target-center 0.3
```

Important: do not pass only `--target-center 0.3` while leaving the
default band `[0.5, 1.5]`. If you do that, the verification calibration
will mark ratio around 0.3 as out of band. The band and center must be
changed together.

Interpretation after training:

- If direct drift improves and R35/K-diversity stay healthy, R41 is
  promising.
- If direct drift barely moves but dynamics/diversity stay healthy,
  do not declare failure. Try a second round with a stronger center,
  for example `target_center=0.6`, on the healthiest cells.
- If direct drift improves but diversity collapses, down-rank or reject
  that cell even if PB1 direct drift looks good.

---

## 3. Verdict on Q2: Cap A3, Do Not Run 14.52x First

Do **not** launch A3 with `w_total=14.52` in the first run.

The A3 recommendation is huge because the current V8/V6 warm-start
already has very small `l_pos_full`:

- A3 `l_pos_full = 0.0106`
- A3 actual grad ratio = `0.069`

The calibration is doing the mathematically correct linear rescale, but
the single-batch estimate does not tell us how the ratio evolves during
training.

Add a cap:

```bash
--max-w-total 5.0
```

For the first nudge calibration with `target_center=0.3`, A3's
recommended value is about `4.36`, which is under the cap and acceptable.
For any later stronger run, A3 should not silently jump to 14.52.

Implementation detail:

- `round41_cascade_calibration.py` should report both:
  - `recommended_w_total_uncapped`
  - `recommended_w_total`
  - `capped: true/false`
  - `max_w_total`

If a cap is applied, the Markdown should show this clearly in the status
column, e.g. `capped`.

Small correction to the concern in the verdict document:

- `5 * 14.52 * 0.05 = 3.63`
- `1.0 * 14.52 = 14.52`

So the L_pos term would still be smaller than the motion MSE term in
that hypothetical. The real risk is not "L_pos exceeds motion MSE"; the
real risk is that total cascade gradient becomes too large and drifts
relative to Stage-1 self loss during training.

---

## 4. Verdict on Q3: Make A0 Calibration-Aware

Do not leave A0 as "P0 crashed". That is misleading.

Use both fixes below:

### 4.1 Calibration script skips control cells

In `round41_cascade_calibration.py`, before invoking P0 for a cfg, load
the YAML and detect:

```python
cascade.enabled == False
```

or equivalently all cascade weights are zero.

For such cells:

- do not run P0;
- write a row with:

  ```json
  {
    "control_cell": true,
    "pass": true,
    "ratio_present": false,
    "recommendation": {
      "in_band": true,
      "recommended_w_total": current_w_total,
      "rec_reason": "control cell: cascade disabled, no calibration needed"
    }
  }
  ```

- Markdown status should be `control`.

### 4.2 P0 check 10 is defensive

In `round41_stage1_cascade_p0_diag.py`, update
`check_10_grad_scale_actual_stack`.

At the top of the function, after recording `cascade_weights`, add a
guard:

```python
active_weights = [
    float(cascade_weights.get("w_motion_mse", 0.0)),
    float(cascade_weights.get("w_world_joint_vel", 0.0)),
    float(cascade_weights.get("w_l_pos_full", 0.0)),
    float(cascade_weights.get("w_anchor_joint_pos", 0.0)),
]
if all(w == 0.0 for w in active_weights):
    out["control_cell"] = True
    out["grad_norm_stage1_self"] = None
    out["grad_norm_actual_cascade_weighted"] = 0.0
    out["ratio_actual_cascade_over_self"] = None
    out["recommended_w_total_for_ratio_1"] = float(
        cascade_weights.get("w_total", 1.0)
    )
    out["pass"] = True
    return out
```

This makes P0 robust even if someone calls it manually on A0.

---

## 5. Concrete Code Changes

### 5.1 `round41_cascade_calibration.py`

Required changes:

1. Change the R41 first-round defaults:

   ```python
   DEFAULT_TARGET_MIN = 0.2
   DEFAULT_TARGET_MAX = 0.5
   DEFAULT_TARGET_CENTER = 0.3
   ```

   If you prefer preserving old defaults for backward compatibility,
   keep constants as-is but update server commands to pass explicit
   values. Codex recommendation: make R41 defaults conservative now,
   because this script is specifically for R41.

2. Add:

   ```python
   DEFAULT_MAX_W_TOTAL = 5.0
   ap.add_argument("--max-w-total", type=float, default=DEFAULT_MAX_W_TOTAL)
   ```

3. Update `_recommend_w_total(...)`:

   Inputs:

   ```python
   max_w_total: float | None = DEFAULT_MAX_W_TOTAL
   ```

   Output keys:

   ```python
   recommended_w_total_uncapped
   recommended_w_total
   capped
   max_w_total
   in_band
   exceeds_abort
   rec_reason
   ```

   Logic:

   ```python
   rec_uncapped = current_w_total * (target_center / measured_ratio)
   rec = rec_uncapped
   capped = False
   if max_w_total is not None and rec > max_w_total:
       rec = max_w_total
       capped = True
   ```

4. Make the Markdown table show capped status.

5. Add a control-cell detector and skip P0 for A0.

   Suggested helper:

   ```python
   def _read_cascade_info(cfg_path: Path) -> dict[str, Any]:
       cfg = OmegaConf.load(str(cfg_path))
       casc = cfg.get("cascade", None)
       ...
   ```

   Return:

   ```python
   {
       "enabled": bool,
       "w_total": float,
       "weights": {...},
       "control_cell": bool,
       "pb1_checkpoint": str | None,
   }
   ```

6. If `control_cell` is true, write a row without P0 invocation.

### 5.2 `round41_apply_calibration.py`

Required changes:

1. If `row.get("control_cell")` is true, skip patching and print:

   ```text
   [apply] stage1_r41_a0_cascade_off: skip control cell
   ```

2. If `row["recommendation"]["capped"]` is true, include that in the
   printout so the operator knows the config is intentionally not at
   the target ratio.

### 5.3 `round41_stage1_cascade_p0_diag.py`

Required change:

- Add the defensive zero-weight guard to `check_10_grad_scale_actual_stack`.

Optional but useful:

- In the calibration JSON, include:

  ```json
  "control_cell": true
  ```

  for A0 if P0 is called manually.

### 5.4 `run_round41_stage1_cascade_matrix.sh`

Required changes:

1. Update help/comments to say first-round recommended calibration is:

   ```bash
   --target-min 0.2 --target-max 0.5 --target-center 0.3 --max-w-total 5.0
   ```

2. In the pre-train audit, log `w_total` and any calibration status if
   available.

3. Do not require A0 to have a valid calibration ratio.

No batch-size change is requested here.

### 5.5 Return Document

Update:

`analyses/2026-06-02_r41_return_for_codex.md`

Include:

- these calibration changes;
- exact chosen target band/center/cap;
- re-calibration results after applying;
- final YAML `w_total` values per cell;
- whether any cell was capped;
- confirmation that A0 is treated as control.

---

## 6. Verification Commands

Run locally:

```bash
python -m py_compile \
  scripts/stage_a_generator/round41_cascade_calibration.py \
  scripts/stage_a_generator/round41_apply_calibration.py \
  scripts/stage_a_generator/round41_stage1_cascade_p0_diag.py

bash -n scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh
```

If pytest is available:

```bash
python -m pytest -q \
  tests/test_stage1_init_checkpoint.py \
  tests/test_pb1_loss_helpers.py
```

Add or update a small unit test if convenient:

- `_recommend_w_total(ratio=0.069, center=0.3, max=5)` returns about
  `4.35`, `capped=false`.
- `_recommend_w_total(ratio=0.069, center=1.0, max=5)` returns `5.0`,
  `capped=true`, uncapped about `14.49`.
- control-cell row is skipped by apply.

---

## 7. Server Recalibration Workflow

After committing and pushing the fixes:

```bash
git pull --ff-only origin master
conda activate piano

export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
export ROUND41_GPUS="0,2"
export ROUND41_NUM_PROCESSES=2
export ROUND41_BUCKETS="val"

python -u scripts/stage_a_generator/round41_cascade_calibration.py \
  --target-min 0.2 \
  --target-max 0.5 \
  --target-center 0.3 \
  --max-w-total 5.0 \
  --out-dir analyses/round41_cascade_calibration

python scripts/stage_a_generator/round41_apply_calibration.py \
  --calibration analyses/round41_cascade_calibration/<stamp>.json \
  --apply

python -u scripts/stage_a_generator/round41_cascade_calibration.py \
  --target-min 0.2 \
  --target-max 0.5 \
  --target-center 0.3 \
  --max-w-total 5.0 \
  --out-dir analyses/round41_cascade_calibration
```

The second calibration should show:

- A0: `control`, no P0 crash
- A1/A2/A3/A4: actual grad ratio in `[0.2, 0.5]`, or clearly marked
  capped if a cap prevented reaching the band

Only then launch:

```bash
bash scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh
```

---

## 8. How to Interpret the First R41 Training Result

This first run is not meant to prove the maximum strength of R41. It is
a safe probe.

Success signs:

- Direct Stage-1 -> PB1 drift improves versus V8/V6 baseline.
- R35 OOD dynamics do not collapse:
  - velocity ratios stay close to V8/V6;
  - pelvis/root dynamics do not flatten.
- K-diversity does not collapse.

If direct drift improves but full cascade worsens:

- do not call R41 failed immediately;
- interpret as Stage-1.5 OOD to R41 coarse;
- next step is likely Stage-1.5 retraining on R41-generated Stage-1.

If direct drift barely changes but R35/K-diversity are healthy:

- try a stronger second calibration, e.g. `target_center=0.6`;
- preferably only on the cells that looked healthy in the first run;
- keep `--max-w-total 5.0` or raise cautiously after inspecting ratio
  drift in metrics.

If dynamics/diversity collapse:

- do not increase target center;
- inspect which term/cell caused collapse.

---

## 9. Final Commit / Push

After implementation and verification:

```bash
git status --short
git add \
  scripts/stage_a_generator/round41_cascade_calibration.py \
  scripts/stage_a_generator/round41_apply_calibration.py \
  scripts/stage_a_generator/round41_stage1_cascade_p0_diag.py \
  scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh \
  analyses/2026-06-02_r41_return_for_codex.md

git commit -m "Tune R41 calibration defaults and control handling"
git push
```

If this instruction document is to be committed too, remember that
`analyses/` is ignored in this repo, so use:

```bash
git add -f analyses/2026-06-02_r41_calibration_next_steps_for_claude.md
```

