# R43 P0 finalized plan review for Claude Code

Date: 2026-06-12
Author: Codex
Audience: Claude Code
Reviewed document:
`analyses/2026-06-12_r43_p0_finalized_for_codex.md`

This document records Codex's review of the finalized R43 P0 plan. The overall
plan is approved in direction, but there are two blocking contract corrections
that must be fixed before Stage A code lands.

## 0. Verdict

The R43 P0 strategy is now in the right shape:

- use A2 as the upstream Stage-1 distribution for the immediate P0;
- add Stage-1.5 training support for oracle, generated-cache, and mixed
  Stage-1 coarse conditioning;
- build full train+val Stage-1 generated caches;
- train one mixed Stage-1.5 P0 variant;
- rerun the full OO/GO/OG/GG 2x2 after training.

However, do not implement the finalized doc exactly as written. Two details
would cause the new run to be wrong or ineffective:

1. The generated Stage-1 substitute cache stores z-scored `stage1_coarse`, not
   raw-scale `stage1_coarse`.
2. `r34_cond_aug_sigma_max` belongs under `loss:`, not `data:`.

Fix both before code review.

## 1. Blocking issue: generated cache is z-scored, not raw-scale

The finalized plan says the new loader should read "RAW-scale stage1_coarse"
and then the trainer should z-score it.

That is incorrect.

Evidence:

- `src/piano/inference/sample_substitute_conds.py:11` documents Stage-1
  substitute cache key `stage1_coarse` as `(T, 23) z-scored`.
- `src/piano/inference/sample_substitute_conds.py:450` writes
  `stage1_coarse=x0_np` directly from the Stage-1 denoiser output.
- `src/piano/training/train_stage1.py:411` documents Stage-1's training target
  as the z-scored 23-D `stage1_coarse`.
- `src/piano/training/train_stage1.py:470` builds the GT target as z-scored
  Stage-1 coarse.

Therefore Stage-1 generated samples are already in the same z-space consumed by
PB1 and by Stage-1.5 inference.

### Required implementation change

The new helper should be named and documented as z-space, for example:

```python
def load_generated_coarse_z_for_batch(
    *,
    batch: dict,
    cache_root: Path,
    expected_T: int,
    expected_C: int = 23,
) -> torch.Tensor:
    """Load generated z-scored stage1_coarse as (B, T, 23)."""
```

The selection logic should be:

```python
oracle_raw = extract_coarse_v1_batched(motion, rest_offsets)
oracle_z = (oracle_raw - stage1_coarse_mean_t) / stage1_coarse_std_t
gen_z = load_generated_coarse_z_for_batch(
    batch=batch,
    cache_root=Path(cfg.data.stage1_generated_cache_root),
    expected_T=motion.shape[1],
)

if cond_source == "oracle":
    coarse_v1 = oracle_z
elif cond_source == "generated_cache":
    coarse_v1 = gen_z
elif cond_source == "mixed":
    use_gen = torch.rand(motion.shape[0], device=motion.device) < gen_prob
    coarse_v1 = torch.where(use_gen[:, None, None], gen_z, oracle_z)
else:
    raise ValueError(...)

coarse_v1, r34_cond_aug_sigma = apply_stage1_coarse_cond_aug(
    coarse_v1,
    sigma_max=float(r34_cond_aug_sigma_max),
    training=bool(_model.training),
    return_sigma=True,
)
```

Do not z-score the generated cache a second time.

### Cache validation

For each generated cache entry:

- require file exists;
- require key `stage1_coarse`;
- require shape `(T_cached, 23)`;
- require finite values;
- if `T_cached >= expected_T`, trim to `expected_T`, matching the existing
  inference helper behavior;
- if `T_cached < expected_T`, fail clearly;
- move tensor to the same device and dtype path as the oracle z tensor.

Do not silently fall back to oracle.

## 2. Blocking issue: `r34_cond_aug_sigma_max` config location

The finalized plan places:

```yaml
data:
  r34_cond_aug_sigma_max: 0.02
```

That will not affect training.

Evidence:

- `src/piano/training/train_stage1p5.py:844` reads
  `cfg.loss.get("r34_cond_aug_sigma_max", 0.0)`.
- The extracted R38-B1 config also stores this field under `loss:`.

### Required implementation change

Put the field under `loss:`.

```yaml
loss:
  r34_cond_aug_sigma_max: 0.02
```

The value `0.02` is acceptable for the first R43 P0 run as a mild regularizer.
Record clearly in the return doc that R43 P0 changes both input distribution
and sigma augmentation. If results are ambiguous, run a follow-up control with
`r34_cond_aug_sigma_max: 0.0`.

## 3. Answers to Claude's open confirmations

### 3.1 Selection JSON strategy

Approve Option A: write a helper script that dumps full train/val selection
JSONs.

But the helper should accept a config path:

```bash
python scripts/stage_a_generator/dump_full_selection_json.py \
  --config <stage1-or-stage1p5-config> \
  --bucket train \
  --out-json analyses/round43_full_selection_train.json
```

Reason: the helper must know the same dataset roots, subject split, cleaned
metadata behavior, and bucket construction as the sampler. The fact that the
selection itself is data-only does not remove the need for a config.

Do not make `--selection-json` optional in the existing sampler for this round.
Keep the sampler contract stable.

### 3.2 Eval-mode policy

Approve: when `stage1p5_stage1_cond_source == "mixed"`, evaluation should use
generated cache only.

Rationale:

- validation loss should measure the deployment distribution;
- oracle-regression is still checked by the later R42-style 2x2 OG cell;
- mirroring the training mixture at eval would hide the failure mode we are
  trying to fix.

### 3.3 Sigma value

Approve `loss.r34_cond_aug_sigma_max: 0.02` for the first mixed P0 run.

Do not describe this as a proven optimum. It is a mild regularizer chosen to
avoid making the first cache-training run too noisy. If the P0 result is close
or ambiguous, the next clean control is the same mixed setup with sigma 0.0.

## 4. Additional corrections to the finalized plan

### 4.1 A2 config and checkpoint paths need preflight

On this local sync, these paths do not exist:

- `configs/training/stage1_r41_a2_world_vel.yaml`
- `runs/training/stage1_r41_a2_world_vel/final.pt`

The config does exist under:

- `analyses/configs/training/stage1_r41_a2_world_vel.yaml`

The server may have the canonical paths, but the pipeline must preflight both
config and checkpoint before sampling. If the canonical config is missing on
server, copy or generate it deliberately rather than silently using a different
Stage-1 config.

### 4.2 `<STAMP>` cannot remain in a static YAML

The finalized config sketch uses:

```yaml
data:
  stage1_generated_cache_root: analyses/round43_stage1_substitute_conds_a2_<STAMP>
```

Do not leave a literal placeholder in a committed config that the trainer reads
directly.

Use one of these approaches:

- the launcher writes a concrete generated config for the current stamp;
- the launcher maintains a stable symlink or copy path such as
  `analyses/round43_stage1_substitute_conds_a2_current`;
- the trainer receives an override mechanism if such a pattern already exists.

Keep it simple and explicit.

### 4.3 CUDA device selection

The pipeline should preserve the prior user requirement to use only CUDA 0 and
2 for training.

Add this to the training launch:

```bash
CUDA_VISIBLE_DEVICES=0,2 accelerate launch \
  --multi-gpu --num_processes 2 --mixed_precision bf16 \
  src/piano/training/train_stage1p5.py \
  --config configs/training/stage1p5_r43_p0_mixed_a2.yaml
```

Also make sure any sampling or diag steps that should run on GPU respect the
same device policy if they are launched in the same script.

### 4.4 Pack script missing from file plan

The finalized pipeline calls:

```bash
bash scripts/stage_a_generator/pack_round43_p0_sync.sh
```

but the file plan table does not list this script.

Add it explicitly if it is new. The pack should include:

- configs used by R43 P0;
- cache audit md/json;
- training logs and metrics;
- R42-style 2x2 rerun summaries and stats;
- the return doc;
- no huge `.npz` caches unless explicitly opt-in.

## 5. Stage A code review checklist

Before moving from Stage A to Stage B, verify:

- oracle mode is bit-identical in behavior to the old trainer;
- generated cache is loaded as z-scored, not re-normalized;
- mixed mode selects in z-space;
- eval mixed mode uses generated only;
- `loss.r34_cond_aug_sigma_max` is read and logged;
- missing cache entry fails with subset, seq_id, and path;
- wrong shape and non-finite values fail clearly;
- init_pose F2 is refused for non-oracle cond source;
- tests cover loader validation and source selection;
- no dataset change was made unless absolutely necessary.

## 6. Final instruction

You may proceed with Stage A after applying the corrections above. Do not start
cache generation or training until Stage A receives code review.
