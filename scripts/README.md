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

- `stage_b_generator/k_sample_oracle.py`: no-retrain diagnostic that samples K
  full-condition variants per fixed validation clip, scores each with the
  existing contact-distance metric, and reports single-sample versus best-of-K
  contact. Use this before adding new Stage B training runs.
