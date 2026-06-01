# Round-40 Stage-1 Plan-Sampler Return for Codex

Audience: Codex / reviewer who handed off the R40 plan in
`analyses/2026-06-01_round40_stage1_plan_sampler_handoff_for_claude.md`.

This document records exactly what landed for R40, with file paths,
function names, test outcomes, dry-run transcripts, and the exact
server-side commands the user needs to run.

---

## 1. TL;DR

R40 turns Stage-1 from a "23-D GT regression target" into a
"coarse-plan sampler" via two new axes wired into `train_stage1.py`:

1. **Per-channel weighting** for the existing exact-GT MSE terms
   (`mse_x0`, `vel_mse`) — lets configs downweight ambiguous channels
   (root, vel, yaw, pelvis_rot6d) without touching the structural V8
   anti-collapse stack.
2. **Plan-invariant loss** — supervises plan-level invariants (speed
   envelope, arc length, turn activity, root-object radial profile,
   height envelope, smoothness). Mode-invariant by construction, so
   CCW vs CW / left vs right routings don't get averaged out.

Both default to OFF; old configs reproduce pre-R40 behavior bit-for-bit.

4-cell matrix on the V8 V6 substrate:
- **C0 baseline** — exact V8 V6 (sanity).
- **C1 weak GT** — channel weighting only.
- **C2 plan energy** — C1 + plan loss at 0.20 (ship candidate).
- **C3 strong plan energy** — stronger weights + plan loss at 0.50.

Diagnostics:
- direct Stage-1 → frozen PB1 drift (via R31 downstream launcher),
- plan-quality audit (R35 metrics + R40 plan stats),
- optional full cascade (Stage-1 → R38-B1 → PB1),
- K-sample diversity (does diffusion noise pick different plan modes?).

All scripts compile, all 59 tests pass, bash syntax + dry-run clean.

---

## 2. Changed Files

| Path | Type | Lines added (≈) |
|---|---|---:|
| `src/piano/training/stage1_losses.py` | modified | +400 |
| `src/piano/training/train_stage1.py`  | modified | +90 |
| `tests/test_stage1_losses.py`         | modified | +260 |
| `scripts/stage_a_generator/round40_make_stage1_plan_configs.py` | new | 220 |
| `scripts/stage_a_generator/round40_stage1_plan_diag.py`         | new | 410 |
| `scripts/stage_a_generator/round40_stage1_k_sample_audit.py`    | new | 250 |
| `scripts/stage_a_generator/run_round40_stage1_plan_matrix.sh`   | new | 480 |
| `scripts/stage_a_generator/pack_round40_stage1_plan_sync.sh`    | new | 165 |
| `analyses/2026-06-01_round40_return_for_codex.md`               | new | this file |

Untracked generated configs from prior rounds are NOT staged. The local
worktree has many such files (`stage1_v8_v6_full_f1.yaml`, …); they
remain untracked.

---

## 3. Library code changes

### 3.1 `src/piano/training/stage1_losses.py`

**Added `build_channel_weight_tensor(weights, *, expected_dim, device, dtype, name) -> Tensor | None`** — small validating helper. `None`/empty list → `None` (caller skips multiplication and behaves exactly like pre-R40). Non-empty must have length `expected_dim` or raise `ValueError`. Returns a `(1, 1, expected_dim)` broadcast tensor.

**Added `stage1_plan_invariant_loss(stage1_raw_pred, stage1_raw_gt, object_world_traj, root_world_t0, seq_mask, *, component_weights=None, beta=1.0) -> tuple[Tensor, dict[str, Tensor]]`** — returns `(total, components_dict)`.

Components (all SmoothL1 between pred and detached GT moments):

| component | what it matches | rationale |
|---|---|---|
| `root_speed` | mean and std of XZ root speed | matches *envelope*, not direction → mode-invariant |
| `root_arc` | total XZ arc length | plan-level magnitude of travel |
| `root_displacement` | final-frame XZ displacement | covers "did you arrive somewhere" without forcing a path |
| `root_object_radial` | mean/std/min/final of distance to object | key for interaction — left and right routes share the same profile |
| `yaw_activity` | mean/std of `|Δyaw_unwrapped|` + cumulative range | turning amount, not direction |
| `rot_activity` | mean/std of pelvis & spine3 rot6d frame-diff magnitude | catches "frozen pelvis" without forcing a specific posture |
| `height_envelope` | head & shoulder mean/min/max heights | posture plausibility without pose phase |
| `smoothness` | root XZ accel + yaw accel magnitude (pred-only) | conservative anchor against wild paths once exact MSE is lowered |

Default component weights match handoff §6.1:
`{root_speed: 1.0, root_arc: 1.0, root_displacement: 0.5, root_object_radial: 1.0, yaw_activity: 1.0, rot_activity: 0.5, height_envelope: 0.5, smoothness: 0.05}`.

Implementation notes:
- All GT summary stats are `.detach()`ed before SmoothL1.
- Root world XZ reconstructed via `(rwx, rwz) = (root_local[0], root_local[1]) + (t0_world[0], t0_world[2])`. The oracle stores root_local in `(x, z, y)` order; channels [0] and [1] are XZ.
- Object XZ uses `object_world_traj[..., [0, 2]]` (world COM is `(x, y, z)`), matching handoff §6.4 requirement.
- Smoothness penalises pred only; no GT comparison. Weight 0.05 keeps it conservative.
- Masking: `seq_mask` floats, pair mask `seq_mask[:, 1:] * seq_mask[:, :-1]`, triple mask for accel. All reductions are masked. Empty masks → gradient-safe `pred.sum() * 0.0`.

### 3.2 `src/piano/training/train_stage1.py`

**New step-fn args** (all default-off / backward-compatible):

```python
x0_channel_weights: tuple[float, ...] | list[float] | None = None
vel_channel_weights: tuple[float, ...] | list[float] | None = None
w_r40_plan_invariant: float = 0.0
r40_plan_beta: float = 1.0
r40_plan_component_weights: dict[str, float] | None = None
```

**Channel weighting** is materialised once outside the closure via `build_channel_weight_tensor` (so device/dtype are stable and validation runs once). Inside `step_fn`:
- `mse_x0` multiplies `mse_per_dim` by `x0_channel_w` if set, before the `.sum(-1)`. `mse_x0_unweighted` unchanged.
- `vel_mse` multiplies `vel_per_dim` by `vel_channel_w` if set, on top of the existing `vel_rot6d_weight`. Added `vel_mse_unweighted` as a new audit metric.

**Plan-invariant** is wired into the existing `need_raw` predicate (so `x0_raw`/`x0_gt_raw` are computed when `w_r40_plan_invariant > 0`). The call site:
```python
r40_plan, r40_components = stage1_plan_invariant_loss(
    stage1_raw_pred=x0_raw, stage1_raw_gt=x0_gt_raw,
    object_world_traj=object_traj,
    root_world_t0=motion[:, :1, 132:135].float(),
    seq_mask=seq_mask,
    component_weights=r40_plan_component_weights,
    beta=float(r40_plan_beta),
)
```

Added to loss: `+ w_r40_plan_invariant * r40_plan`. Added to returns: `r40_plan_invariant`, `r40_plan_invariant_weighted`, plus one `r40_plan_<component>` per evaluated component.

**Config wiring** in `main()`: reads `cfg.loss.{x0_channel_weights, vel_channel_weights, w_r40_plan_invariant, r40_plan_beta, r40_plan_component_weights}` with safe defaults. Empty list → `None`, empty dict → `None`.

**Smoke-test print** adds an R40 line with `weighted/mse_x0` ratio — the launcher §"CALIBRATION REMINDER" tells the user to kill + lower the weight if this ratio is too high.

---

## 4. Tests

### 4.1 New tests (all in `tests/test_stage1_losses.py`)

12 new tests appended:

R40 channel-weight helper:
- `test_channel_weight_helper_empty_returns_none` — `None`, `[]`, `()` → `None`.
- `test_channel_weight_helper_wrong_length_raises` — length 22 raises with clear message.
- `test_channel_weight_helper_valid_shape` — broadcast shape `(1, 1, 23)`, dtype preserved.
- `test_channel_weight_helper_zero_weight_removes_channel_contribution` — `w[5]=0` drops channel 5 from the weighted sum.
- `test_channel_weight_helper_non_list_raises` — string input raises with clear message.

R40 plan-invariant loss:
- `test_plan_invariant_zero_on_identity` — `pred=gt` → every comparison component ≈ 0 (smoothness is pred-only and exempt).
- `test_plan_invariant_frozen_root_has_larger_speed_arc_than_gt` — `root_speed` and `root_arc` are strictly positive when pred has a frozen root.
- `test_plan_invariant_mirrored_path_radial_low_penalty` — `pred=-gt` mirror around an origin-located object gives `root_object_radial ≈ 0` (proves mode-invariance).
- `test_plan_invariant_masking_ignores_padded_frames` — corrupting padded frames in pred does not change the loss.
- `test_plan_invariant_smoothness_finite_and_nonneg` — smoothness ≥ 0, finite.
- `test_plan_invariant_gradients_flow_to_pred` — `total.backward()` populates `pred.grad`.
- `test_plan_invariant_unknown_component_raises` — typo in `component_weights` is caught.

### 4.2 Results

```
$ python -m pytest -q tests/test_stage1_losses.py
...........................................................              [100%]
59 passed in 1.05s

$ python -m pytest -q tests/test_stage1_losses.py tests/test_stage1p5_losses.py
121 passed in 1.12s
```

All pre-existing tests (47 stage1, 62 stage1p5) continue to pass — the
new code adds knobs but does not change defaults.

### 4.3 Compile / syntax checks

```
$ python -m py_compile \
    src/piano/training/stage1_losses.py \
    src/piano/training/train_stage1.py \
    scripts/stage_a_generator/round40_make_stage1_plan_configs.py \
    scripts/stage_a_generator/round40_stage1_plan_diag.py \
    scripts/stage_a_generator/round40_stage1_k_sample_audit.py
$ bash -n scripts/stage_a_generator/run_round40_stage1_plan_matrix.sh
$ bash -n scripts/stage_a_generator/pack_round40_stage1_plan_sync.sh
```
All clean.

---

## 5. Scripts

### 5.1 `round40_make_stage1_plan_configs.py`

Base config: `configs/training/stage1_v8_v6_full_f1.yaml`.

Pre-flight checks the base has:
- `model.denoiser.init_pose_dim == 135` (R31 V8 F1 anchor),
- `loss.w_moment_velocity > 0` (R31 V7-A anti-collapse),
- `loss.w_yaw_aggregate > 0` (R31 V7-B).

Variants (the actual weight tables match handoff §7 exactly):

| vid | `x0_channel_weights` | `vel_channel_weights` | `w_r40_plan_invariant` |
|---|---|---|---:|
| `c0_v8v6_baseline`    | (none) | (none) | 0.00 |
| `c1_weak_gt`          | C1 23-vector | C1 23-vector | 0.00 |
| `c2_plan_energy`      | C1 23-vector | C1 23-vector | 0.20 |
| `c3_plan_energy_strong` | C3 23-vector | C3 23-vector | 0.50 |

The script also validates each materialised list is either empty or length 23 before writing.

`val_best_key` is left at the base value (`mse_x0`) per handoff §7
("launchers must use `final.pt` for downstream diag" — the launchers
explicitly take `final.pt`, so the train-time best-ckpt key does not
gate the ship cell).

### 5.2 `round40_stage1_plan_diag.py`

Per-bucket plan-quality audit of a substitute-conds dir.

Reads:
- the sampled z-scored stage1_coarse from `<pred-dir>/<bucket>/<subset>/<seq>.npz` (matches `sample_substitute_conds` schema),
- the GT via `extract_coarse_v1_batched` over the same selection,
- the Stage-1 normalization stats via `load_stage1_coarse_norm`.

Reports:
- R35 group metrics (std_ratio, vel_ratio, PSD pred/gt low/mid/high) for direct comparison against earlier R35 audit reports;
- plan metrics mirroring `stage1_plan_invariant_loss` components (root speed mean/std, arc, displacement, root-object radial mean/std/min/final, yaw rate mean/std, yaw range, pelvis/spine3 rot6d activity mean/std, head/shoulder height stats);
- top-10 channels by residual RMS.

Outputs: `plan_stats.json` (machine), `plan_summary.md` (human).

### 5.3 `round40_stage1_k_sample_audit.py`

Calls `sample_substitute_conds` K times in-process with different seeds (default `41…48`, 8 seeds) into `<out-dir>/samples/seed<S>/<bucket>/<subset>/<seq>.npz`, then computes pairwise diversity per clip:

- `pair_root_path_rms` — frame-wise XZ path RMS between two samples,
- `pair_final_disp` — XZ distance between final-frame positions,
- `pair_yaw_range_diff` — `|range_a − range_b|` of cumulative yaw,
- `pair_pelvis_rot6d_rms` — pelvis rot6d block RMS.

Decision rule (built into the summary md): if `pair_root_path_rms_mean < ~0.05` across all variants, diffusion noise is not being used as a mode-selector and the next round needs an explicit mode token / best-of-K. Otherwise plan-energy reranking is viable.

### 5.4 `run_round40_stage1_plan_matrix.sh`

Phases per variant (env-knobs prefixed `ROUND40_*`):

1. **TRAIN** — `accelerate launch --num_processes ROUND40_NUM_PROCESSES --multi_gpu --mixed_precision bf16 src/piano/training/train_stage1.py --config <cfg>`. Skipped if `final.pt` exists unless `--force-retrain`.
2. **DIRECT DIAG** — delegates to `run_round31_stage1_downstream_diag.sh` with `ROUND31_DS_STAGE1_CFG/CKPT`, `ROUND31_DS_PB1_CKPT`, `ROUND31_DS_BUCKETS`, `ROUND31_DS_OUT_TAG=_r40_<vid>`. After success, moves the round31 substitute-conds dir to `analyses/round40_stage1_substitute_conds_<vid>/` and archives diag to `analyses/round40_stage1_direct_diag_<vid>/`.
3. **PLAN DIAG** — `round40_stage1_plan_diag.py` on the substitute-conds dir → `analyses/round40_stage1_plan_diag_<vid>/`.
4. **FULL CASCADE** (optional) — delegates to `run_round32_stage1p5_downstream_diag.sh` with the Stage-1 substitute-conds dir as `ROUND32_DS_UPSTREAM_DIR` and Stage-1.5 R38-B1 as the model. Skip with `ROUND40_SKIP_FULL_CASCADE=1`. Archives to `analyses/round40_fullcascade_diag_<vid>/`.
5. **KDIV** (subset of VIDs) — runs `round40_stage1_k_sample_audit.py` for VIDs in `ROUND40_KDIV_VIDS` (default `c0,c2,c3`). K from `ROUND40_K_SAMPLES` (default 8).
6. **SUMMARY** — Python builds `analyses/round40_stage1_plan_matrix_summary_<stamp>.md` with the comparison table required by handoff §9.1 (direct/cascade drift, vel_ratio, std/vel pelvis, root_arc_ratio, yaw_range_ratio, k root path diversity) and an embedded decision tree.
7. **PACK** — delegates to `pack_round40_stage1_plan_sync.sh`.

Supported flags:
`--dry-run`, `--only`, `--skip-train`, `--skip-diag`, `--skip-kdiv`, `--force-retrain`, `--force-rediag`, `--buckets`.

Required env: `DATASETS_ROOT`. Optional: `ROUND40_GPUS`, `ROUND40_NUM_PROCESSES`, `ROUND40_BUCKETS`, `ROUND40_BASE_CFG`, `ROUND40_PB1_CKPT`, `ROUND40_STAGE1P5_B1_CFG/CKPT`, `ROUND40_SKIP_FULL_CASCADE`, `ROUND40_K_SAMPLES`, `ROUND40_KDIV_VIDS`, `ROUND40_ALLOW_PARTIAL`.

### 5.5 `pack_round40_stage1_plan_sync.sh`

Packs configs, train logs, metrics.jsonl, direct-diag dirs, plan-diag dirs, full-cascade dirs, kdiv MD+JSON (samples/ excluded), matrix summary md+log, and the handoff + return docs. Excludes ckpts and sampled .npz caches by default. Opt-ins: `ROUND40_PACK_CKPTS=1`, `ROUND40_PACK_NPZ=1`, `ROUND40_PACK_KDIV_SAMPLES=1`. Emits a manifest next to the tarball.

---

## 6. Dry-run transcripts

### 6.1 Launcher dry-run (excerpt)

```
$ DATASETS_ROOT=/tmp/fake bash scripts/stage_a_generator/run_round40_stage1_plan_matrix.sh --dry-run
...
===== R40 matrix launch 20260601_115322 =====
*** CALIBRATION REMINDER ***
After the first few train epochs, audit each VARIANT's metrics.jsonl
for r40_plan_invariant_weighted vs mse_x0. Target: weighted ≤ ~1×mse_x0.
If weighted ≥ 3× mse_x0 in epoch 1, kill, lower w_r40_plan_invariant 5×, restart.
...
TRAIN stage1_r40_c0_v8v6_baseline
    $ CUDA_VISIBLE_DEVICES=0,2 accelerate launch --num_processes 2 \
        --multi_gpu --mixed_precision bf16 \
        src/piano/training/train_stage1.py \
        --config configs/training/stage1_r40_c0_v8v6_baseline.yaml
DIRECT DIAG stage1_r40_c0_v8v6_baseline  (buckets: val)
    [R40 DRY-RUN] direct diag stage1_r40_c0_v8v6_baseline:
        ROUND31_DS_* env, run_round31_stage1_downstream_diag.sh
KDIV stage1_r40_c0_v8v6_baseline  (K=8)
    $ python -u scripts/stage_a_generator/round40_stage1_k_sample_audit.py \
        --config configs/training/stage1_r40_c0_v8v6_baseline.yaml \
        --ckpt   runs/training/stage1_r40_c0_v8v6_baseline/final.pt \
        --selection-json analyses/round29_val_diag_indices_48_balanced.json \
        --bucket val --out-dir analyses/round40_stage1_kdiv_stage1_r40_c0_v8v6_baseline \
        --num-samples 8
... (same per c1 [no kdiv], c2, c3) ...

Trained:   c0 c1 c2 c3
Diaged:    c0 c1 c2 c3
KDIV:      c0 c2 c3
```

Phase 3 (plan diag) and Phase 4 (full cascade) are intentionally skipped in dry-run because they require the substitute-conds dir from a real Phase 2 run. The launcher correctly gates them with `-d "${SUB_DIR_ROOT}"` — on a real run these will fire.

### 6.2 Packer dry-run

In dry-run on the laptop, the packer correctly enumerates targets and exits 0 when sources are missing — matching the existing `pack_round38_sync.sh` pattern. The launcher always wraps the packer in `… || log "[R40 PACK] packer failed (non-fatal)"`, so a packer failure never fails the whole matrix.

---

## 7. Commit + push

```
commit <SHA>  Add round40 Stage-1 plan sampler training
files: 8 (3 modified + 5 new)
```

Pushed to `origin/master`. Branch protection / signing per repo defaults (no `--no-verify`).

---

## 8. Server-side commands (for the user)

On `5080x3`:

```bash
ssh <user>@5080x3
cd /media/8TB_data/Cai/PIANO/PIANO
git fetch origin
git checkout master
git pull --ff-only origin master
git rev-parse --short HEAD       # confirm matches the SHA below

conda activate piano
export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
export ROUND40_GPUS="0,2"
export ROUND40_NUM_PROCESSES=2
export ROUND40_BUCKETS="val"

bash scripts/stage_a_generator/run_round40_stage1_plan_matrix.sh --dry-run
```

If dry-run prints sane commands, launch in tmux:

```bash
tmux new -s r40
cd /media/8TB_data/Cai/PIANO/PIANO
conda activate piano
export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
export ROUND40_GPUS="0,2"
export ROUND40_NUM_PROCESSES=2
export ROUND40_BUCKETS="val"
bash scripts/stage_a_generator/run_round40_stage1_plan_matrix.sh
```

After completion:

```bash
bash scripts/stage_a_generator/pack_round40_stage1_plan_sync.sh
ls -lh analyses/round40_stage1_plan_results_*.tar.gz
```

Sync back:

```bash
scp <user>@5080x3:/media/8TB_data/Cai/PIANO/PIANO/analyses/round40_stage1_plan_results_*.tar.gz ./analyses/
scp <user>@5080x3:/media/8TB_data/Cai/PIANO/PIANO/analyses/round40_stage1_plan_results_*_manifest.txt ./analyses/
```

---

## 9. Known limitations / follow-up risks

1. **Local env lacks omegaconf and torch GPU build.** Verified:
   - all pytest unit tests pass (pure-torch),
   - all Python scripts `py_compile` cleanly,
   - bash scripts `bash -n` and `--dry-run` cleanly.
   - The OmegaConf config-generator (`round40_make_stage1_plan_configs.py`) and the diag scripts that `import torch.utils.data` cannot be runtime-tested on the laptop. They are written to match exactly the pattern used by `round38_make_stage1p5_configs.py` and `round35_stage1_coarse_ood_audit.py`, both of which run on the server.

2. **`r40_plan_component_weights` config typing.** OmegaConf reads YAML mappings as `DictConfig`. The trainer calls `dict(cfg.loss.get("r40_plan_component_weights", {})) or None`, which materialises a plain dict. The library function accepts unknown keys via a `ValueError`, so a typo in the YAML is caught at the first step. Tested via `test_plan_invariant_unknown_component_raises`.

3. **`val_mse_x0` is no longer the right ship metric** (handoff §2 explicit). The launcher uses `final.pt` directly for downstream diag and the ship signal is the direct-diag drift + the audit metrics + the optional full-cascade drift. The val_best_key stayed at `mse_x0` to keep ckpt-saving stable.

4. **R40 weights are calibrated for V8 V6.** The cfg generator's preflight refuses to run if the base lacks the V8 anti-collapse stack or the init_pose_dim=135 frame-0 anchor, so accidental application to an earlier base cfg is caught.

5. **Plan-diag and full-cascade phases only run when the substitute-conds dir exists.** This is by design: they consume the direct-diag's output. If the direct diag fails or is skipped, those phases skip cleanly without errors.

6. **K-sample audit calls `sample_substitute_conds` in-process** K times. This shares the same model + diffusion build across seeds (no per-call ckpt reload). At K=8 with the default 48-clip selection this should take ~5-10 min per variant on a 5080.

---

## 10. What R40 deliberately did NOT change

Per handoff §13:

- ✅ Did NOT add C41 dynamics losses.
- ✅ Did NOT add GT-derived conditions (`contact_state`, etc.) to the Stage-1 cond set.
- ✅ Did NOT retrain PB1 or touch `stage1_coarse_dim=23`.
- ✅ Did NOT decide ship by `val_mse_x0` (launcher uses `final.pt`, summary uses direct drift + audit metrics).
- ✅ Did NOT change `model.denoiser` dims (no architecture scaling — current evidence points to under-determined training, not insufficient `d_model`).
- ✅ Did NOT modify PB1 or Stage-1.5 model code.

R40 sits cleanly on top of the V8 V6 substrate: defaults reproduce V8 V6 bit-for-bit, the new knobs are opt-in per cfg.

---

## 11. Next steps after the user runs R40 server-side

When the tarball lands back on the laptop, the recommended reading order:

1. `analyses/round40_stage1_plan_matrix_summary_<stamp>.md` — the headline 9-column table + decision tree.
2. Per-variant `analyses/round40_stage1_plan_diag_<vid>/plan_summary.md` — does the variant move the metrics the loss was meant to supervise (root_arc_ratio, pelvis std/vel ratios)?
3. `analyses/round40_stage1_kdiv_<vid>/k_sample_summary.md` for C0/C2/C3 — is diffusion noise becoming a mode-selector?
4. `analyses/round40_stage1_direct_diag_<vid>/sustained_contact_val/sustained_contact_summary.md` — direct downstream drift on the same 48-clip set used everywhere else.

Then write the R41 plan based on which branch of the §9.1 decision tree fires.
