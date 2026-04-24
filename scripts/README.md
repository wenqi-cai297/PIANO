# PIANO Scripts

Operational entry points, grouped by **pipeline stage**.

Library code lives in `src/piano/` (a single functional package — `data/`,
`models/`, `training/`, `inference/`, `evaluation/`, `utils/`, `checks/`).
Stable CLI commands are registered as `console_scripts` in `pyproject.toml`
and resolve against `src/piano/*:main`. This directory holds the ad-hoc
shell wrappers + one-off Python entry scripts that orchestrate those
CLIs (environment setup, multi-subset loops, backup/restore logic, etc).

## Layout

```
scripts/
├── prep/                          env + data bring-up (one-time per server)
│   ├── check_momask_weights.sh
│   ├── download_omomo.sh
│   ├── preprocess_omomo.sh
│   └── preprocess_interact.sh
│
├── stage1_pseudo_labels/          Stage 1 — pseudo-label extraction + QA + vis
│   ├── extract_pseudo_labels_omomo.sh
│   ├── extract_pseudo_labels_interact.sh
│   ├── rerun_pseudo_labels_interact.sh    # backup-and-rerun with fresh config
│   ├── clean_pseudo_labels.py             # post-hoc metadata filter
│   ├── probe_mesh_up_axis.py              # debug helper for sitting gate
│   ├── pseudo_label_stats.sh              # aggregate summary.json numbers
│   ├── threshold_sweep.sh                 # per-body-part distance sweep
│   ├── action_segment_sweep.sh            # (unused — stricter-prior dead end)
│   ├── probe_text_annotations.sh          # (unused — stricter-prior dead end)
│   ├── visualize_pseudo_labels.sh         # single-subset vis trigger
│   ├── visualize_finished_subsets.sh      # batch vis across all subsets
│   └── vis_v9_pseudo_labels.sh            # targeted v9 vis (6 sampling groups)
│
├── stage_a_predictor/             Stage A — interaction predictor training
├── stage_b_generator/             Stage B — motion generator finetune
├── stage_c_joint/                 Stage C — joint finetune + consistency loss
├── eval/                          evaluation pipelines
│
├── checks/                        cross-stage sanity checks (format + smoke)
│   ├── check_hoi_dataset.sh
│   ├── check_interact_format.sh
│   ├── check_omomo_format.sh
│   ├── check_object_convention.sh
│   └── inference_smoke_test.sh
│
└── vis/                           cross-stage visualisation
    └── visualize_motion.sh
```

`stage_a_predictor/`, `stage_b_predictor/`, `stage_c_joint/`, and `eval/`
are empty placeholders for the next phases of work — populated once
Stage A training, generator finetune, etc. come online.

## Where does a given `.sh` vs `.py` live?

The criterion is **role**, not extension:

- `.sh` wrappers that activate an env, set paths, and call a registered
  console script (`piano-<cmd>`) → `scripts/`
- `.py` files that are run directly with `python scripts/<cat>/<name>.py`
  and have `argparse` + `__main__` → `scripts/` too
- Library code imported by other modules → `src/piano/`
- Stable CLI entry points registered in `pyproject.toml` as
  `console_scripts` → `src/piano/<module>.py` with a `main()` function
  (must live under the package for `console_scripts` resolution)

See the project-wide convention note for the full rationale.
