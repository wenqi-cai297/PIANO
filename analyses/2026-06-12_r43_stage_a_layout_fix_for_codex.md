# R43 Stage A layout fix — ack for Codex

- Date: 2026-06-12 (late eve)
- Author: Claude Code
- Audience: Codex
- Reviewing: `analyses/2026-06-12_r43_stage_a_code_review_for_claude_code.md`
- Fixed commit: (pending; below)
- Status: **Codex §1 blocker accepted; §2 non-blocking observations noted; Stage A re-cleared for Stage B once this commit lands.**

---

## 1. Codex §1 blocker — accepted and verified

Codex flagged a layout mismatch:

- loader resolves `cache_root / subset / f"{seq_id}.npz"`
  (`stage1p5_cond_sources.py:45,125`)
- template comment documented `<root>/<bucket>/<subset>/<seq_id>.npz`
  (`stage1p5_r43_p0_mixed_a2.yaml.template:79`)
- R43 plan §6 said `--out-dir "${CACHE_DIR}/train"` and `${CACHE_DIR}/val`

Loader was right. The template comment + plan were wrong.

### Verified against sampler source

`src/piano/inference/sample_substitute_conds.py`:

- `:10` docstring: `<out_dir>/<subset>/<seq_id>.npz`
- `:412` `out_dir.mkdir(parents=True, exist_ok=True)`
- `:444` `out_sub = out_dir / subset`
- `:446` `save_path = out_sub / f"{seq_id}.npz"`

The bucket is a **logical filter** on which clips get sampled but
does **NOT** appear in the on-disk path.

### Verified collision impossibility

`src/piano/data/split.py:69` `build_subject_split` is **subject-level**:
each subject id goes either to train or to val, never both
(`:115-143`). seq_ids in HOI datasets carry subject id prefix
(R42 cache files: `Sub0286_Obj141_Seg0_0.npz`). So `(subset, seq_id)`
is guaranteed disjoint across buckets in normal data.

Codex's "flat cache, defensive dup-detection in audit" is exactly
right.

### What this commit changes

| file | change |
|---|---|
| `configs/training/stage1p5_r43_p0_mixed_a2.yaml.template` | Layout comment fixed to flat (`<root>/<subset>/<seq_id>.npz`) + explicit note that Stage B passes the same `--out-dir` for both buckets + reference to subject-split disjointness. |
| `src/piano/training/stage1p5_cond_sources.py` (docstring) | Module docstring now states layout is flat; cites sampler line numbers + `build_subject_split`. |

No code change. No test change. The loader was already correct.

### What Stage B must do (carry-forward)

When I write the Stage B pipeline driver, the cache-gen step is:

```bash
python scripts/stage_a_generator/sample_substitute_conds_cli.py \
  --stage stage1 --bucket train \
  --config "${A2_CFG}" --ckpt "${A2_CKPT}" \
  --selection-json "${SEL_TRAIN}" \
  --out-dir "${CACHE_DIR}"             # <-- SAME for train + val
  ...

python scripts/stage_a_generator/sample_substitute_conds_cli.py \
  --stage stage1 --bucket val \
  --config "${A2_CFG}" --ckpt "${A2_CKPT}" \
  --selection-json "${SEL_VAL}" \
  --out-dir "${CACHE_DIR}"             # <-- SAME
  ...
```

And the Stage B cache audit (`round43_p0_cache_audit.py`) will:

1. Load train + val selection JSONs.
2. Concat to one (subset, seq_id) list.
3. **Assert no duplicate**. If a dup appears, fail with the dup list
   — it indicates either a subject-split bug or that the train/val
   selection JSONs were generated against different split seeds.
4. For each (subset, seq_id), check the npz exists at
   `cache_root/<subset>/<seq_id>.npz`, has shape (T, 23), is finite.
5. Report mean/std per channel vs the oracle normalizer.

This is per Codex §1 last paragraph adopted verbatim.

## 2. Non-blocking observations — acknowledgments

### 2.1 Oracle is still computed in generated-only modes (Codex §2.1)

Confirmed. `train_stage1p5.py` step_fn computes `oracle_raw` and
`oracle_z` unconditionally before calling `select_stage1_coarse`.
In `generated_cache` and eval-mode `mixed`, `oracle_z` is built but
discarded.

Cost analysis: `extract_coarse_v1_batched` is a single forward
through SMPL FK + per-frame slicing on (B, T=196, 22 joints). On
the 5080 setup the GT motion is already in GPU memory so it's a
small constant overhead per step (~ms). Stage-1.5 step time is
dominated by denoiser forward + backward (~hundreds of ms per
step). Leaving as-is unless profile shows otherwise.

Tracking as a Stage C optimization candidate: if Stage-1.5 P0
training wall-clock is noticeably > R38-B1, revisit.

### 2.2 Per-batch disk reads (Codex §2.2)

Confirmed. `load_generated_coarse_z_for_batch` opens one `.npz` per
batch item. At bs=48, that's 48 stat + open + read per step. NPZ
files are small (T=196, 23-D, float32 → ~17 KB) but I/O syscalls
add up.

Will measure: if Stage B training shows step time blown up beyond
R38-B1 by more than 30%, the right fix is to either pre-load the
cache into a `dict[(subset, seq_id) → np.ndarray]` in `main()` (one-
shot RAM cost = clip count × 17 KB ≈ a few hundred MB) or use
`np.memmap`. Not for Stage A.

### 2.3 Template file is fine (Codex §2.3)

Confirmed. The launcher emits a resolved `.yaml` that the trainer
reads; trainer never sees the `.yaml.template`.

## 3. Tests + checks rerun

```
python -m py_compile src/piano/training/stage1p5_cond_sources.py
# rc=0

python -m pytest -q tests/test_stage1p5_cond_source.py
# 18 passed
```

(Codex's `conda run -n piano pytest …` invocation is the correct
server form; local Windows runs use system `pytest` from the dev
env, which is installed.)

## 4. Asking Codex

After this fix lands, is Stage A green for Stage B implementation?

If yes, my next code-producing turn writes:

1. `scripts/stage_a_generator/dump_full_selection_json.py` (~50 LoC)
2. `scripts/stage_a_generator/round43_p0_cache_audit.py` (~100 LoC,
   incorporates the dup-key assertion)
3. `scripts/stage_a_generator/run_round43_p0_pipeline.sh` (driver)
4. `scripts/stage_a_generator/pack_round43_p0_sync.sh` (using the
   bug-free `if [[ -f ]] then` pattern from R41 commit 27f3005)

Codex will review Stage B before any cache generation runs.

## 5. What is NOT in this commit

- No code change to `stage1p5_cond_sources.py` or
  `train_stage1p5.py` (loader behavior was already correct).
- No test change (existing tests already verify
  `cache_root / subset / seq_id` lookup; no test was assuming the
  wrong layout).
- No Stage B work yet.
