# PIANO Scripts

Operational entry points grouped by pipeline stage.

Rule of thumb:

- importable library code lives in `src/piano/`;
- direct runnable scripts live in `scripts/`;
- stable `console_scripts` registered in `pyproject.toml` point to package
  modules under `src/piano/`.

## Layout

```text
scripts/
  prep/                    environment and data bring-up
  stage1_pseudo_labels/    pseudo-label extraction, QA, visualization
  stage_a_predictor/       Stage A predictor training helpers
  stage_b_generator/       Stage B generator training/eval helpers
  stage_c_joint/           future joint finetune helpers
  eval/                    evaluation wrappers
  checks/                  cross-stage sanity checks
  vis/                     visualization wrappers
```

Both `.sh` and `.py` files can belong in `scripts/` if they are run directly.
A Python file with `argparse` plus `if __name__ == "__main__"` is a script, not
library code.

## Stage B Diagnostics

- `stage_b_generator/run_v13_target_trajectory.sh`: train/eval runner for the
  v13 loss that tracks the named contact body part against its object-local
  contact target trajectory and logs temporal-coupling metrics. Defaults to
  compact wandb CSV / eval JSON output; set `SUMMARY_DETAIL=full` for per-clip
  debug summaries.
- `stage_b_generator/run_v14_sampled_st_contact.sh`: v14 runner for the C2c
  sampled-path variant. It keeps v13's target-trajectory loss but takes decoded
  aux logits from the all-mask MaskGIT/CFG first step and decodes with
  straight-through Gumbel hard codebook samples.
- `stage_b_generator/k_sample_oracle.py`: no-retrain diagnostic that samples K
  full-condition variants per fixed validation clip, scores each sample, and
  saves best-of-K outputs for visualization. Default selection is the existing
  contact-distance metric; `--selection-metric composite` adds a moving-object
  kinematic-coupling penalty for the temporal-binding failure mode.
- `stage_b_generator/measure_temporal_coupling.py`: post-process diagnostic for
  generated runs. Measures whether moving-object frames have a body part that
  is stable in object-local coordinates, reusing the pseudo-label extractor's
  kinematic-coupling criterion. Defaults to compact aggregate JSON; pass
  `--detail full` when the per-clip/worst-case rows are needed.
