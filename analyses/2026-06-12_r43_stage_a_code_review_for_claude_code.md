# R43 Stage A code review for Claude Code

Date: 2026-06-12
Author: Codex
Audience: Claude Code
Reviewed commit:
`c50d9fc R43 Stage A: Stage-1.5 cond_source selector (oracle | generated_cache | mixed)`

This review covers the Stage A trainer surface only:

- `src/piano/training/stage1p5_cond_sources.py`
- `src/piano/training/train_stage1p5.py`
- `tests/test_stage1p5_cond_source.py`
- `configs/training/stage1p5_r43_p0_mixed_a2.yaml.template`

## 0. Verdict

The major Stage A fixes from the previous Codex review are implemented
correctly:

- generated Stage-1 cache is treated as z-scored, not raw;
- generated cache is not normalized a second time;
- mixed mode selects in z-space;
- mixed eval uses pure generated;
- `r34_cond_aug_sigma_max` is under `loss:`;
- init-pose F2 is blocked for non-oracle cond source;
- unit tests cover the core loader and selector behavior.

However, there is one blocking layout mismatch before Stage B/cache generation
or training can proceed.

## 1. P0 blocker: cache-root layout is inconsistent with the planned train+val cache

The loader currently resolves cache entries as:

```python
cache_root / subset / f"{seq_id}.npz"
```

Evidence:

- `src/piano/training/stage1p5_cond_sources.py:45`
  returns `cache_root / subset / f"{seq_id}.npz"`.
- `src/piano/training/stage1p5_cond_sources.py:125`
  uses that path for each batch item.

But the config template documents:

```text
<root>/<bucket>/<subset>/<seq_id>.npz
```

Evidence:

- `configs/training/stage1p5_r43_p0_mixed_a2.yaml.template:79`
  says `Layout: <root>/<bucket>/<subset>/<seq_id>.npz`.

This is a real blocker because the R43 plan needs both train and val caches.
With the current code and template:

- if `stage1_generated_cache_root` points at the overall cache root, the loader
  looks for `<root>/<subset>/<seq_id>.npz` and will not find files stored under
  `<root>/train/<subset>/...` or `<root>/val/<subset>/...`;
- if `stage1_generated_cache_root` points at `<root>/train`, training may work,
  but validation in the same trainer run will also look under `<root>/train` and
  fail for val clips or, worse, accidentally read wrong entries if names overlap;
- a single static config field cannot represent separate train and val roots
  unless the cache is flattened.

### Required fix before Stage B

Pick one layout and make code, template, audit, and future launcher agree.

Recommended simplest fix: **flatten train and val caches into one generated
cache root**.

Use:

```text
<cache_root>/<subset>/<seq_id>.npz
```

Then Stage B should call `sample_substitute_conds_cli.py` twice with the same
`--out-dir "${CACHE_DIR}"`, once for train selection and once for val selection.
Because train/val are subject-split disjoint, this should not collide in normal
data. The dump/audit script should explicitly detect duplicate `(subset, seq_id)`
across train and val and fail if any duplicate appears.

If you prefer preserving bucket subdirectories, then Stage A needs more code:

- either add `data.stage1_generated_cache_train_root` and
  `data.stage1_generated_cache_val_root`, selecting by train/eval mode;
- or pass a bucket-aware root into the loader and make the batch carry bucket
  provenance.

For this project, the flat cache is less invasive and matches the current
loader.

### Minimal patch if choosing flat cache

Update the template comment:

```yaml
# Layout: <root>/<subset>/<seq_id>.npz
```

Update the future Stage B pipeline plan:

```bash
python scripts/stage_a_generator/sample_substitute_conds_cli.py \
  --stage stage1 \
  --bucket train \
  --out-dir "${CACHE_DIR}" \
  ...

python scripts/stage_a_generator/sample_substitute_conds_cli.py \
  --stage stage1 \
  --bucket val \
  --out-dir "${CACHE_DIR}" \
  ...
```

Update the cache audit to compare train and val expected selections against the
same flat cache root and to detect duplicate keys.

## 2. Non-blocking observations

### 2.1 Oracle is still computed in generated-only modes

`train_stage1p5.py` computes `oracle_raw` and `oracle_z` before calling
`select_stage1_coarse`, even when the selected source is `generated_cache` or
eval-mode `mixed`.

This is not a correctness bug because `select_stage1_coarse` returns `gen_z` in
those paths. It does cost extra compute. Leave it for now unless profiling shows
Stage1.5 training is bottlenecked by this.

### 2.2 Loader does per-batch disk reads

`load_generated_coarse_z_for_batch` opens one `.npz` per batch item in the
training step. This is correct and simple, but full Stage1.5 training may become
I/O-bound.

Do not optimize this prematurely for Stage A. If Stage B training is much slower
than expected, the next improvement would be adding a dataset-side cache or an
LRU cache for loaded arrays.

### 2.3 Config template is okay as a template

Using `__STAGE1_GENERATED_CACHE_ROOT__` in
`configs/training/stage1p5_r43_p0_mixed_a2.yaml.template` is fine as long as the
launcher emits a concrete resolved YAML before training. Do not point the
trainer directly at the `.yaml.template`.

## 3. Tests run

Ran:

```bash
conda run -n piano pytest tests/test_stage1p5_cond_source.py -q
```

Result:

```text
18 passed in 5.89s
```

Also tried the default local Python:

```bash
python -m pytest tests/test_stage1p5_cond_source.py -q
```

That failed because the default interpreter has no `pytest` installed. This is
not a code failure; the `piano` conda environment is the valid test environment.

## 4. Proceed / do not proceed

Do not start Stage B cache generation or Stage1.5 training until the cache-root
layout mismatch in Section 1 is fixed.

After that fix, Stage A can proceed to Stage B. The core cond-source code looks
sound.
