# PIANO Compact Specification

PIANO: Physically-Informed Adaptive iNteraction Orchestration.

## Claim

Object-conditioned human motion should not be modeled as only
`text + object -> motion`. PIANO inserts an interpretable temporal interaction
plan:

```text
text + object + initial state -> z_int -> motion
```

`z_int` is meant to capture how the object changes contact timing, contact
target, support, and manipulation phase.

## Data

Primary data track: InterAct-derived subsets:

- `chairs`
- `imhd`
- `neuraldome`
- `omomo_correct_v2`

The project keeps two coordinate views:

- HumanML3D/MoMask `motion_263` features for VQ/MoMask compatibility.
- Raw joints/object transforms for interaction pseudo-label extraction and
  contact measurement.

Use upstream MoMask/HumanML3D recovery utilities when possible; do not
reimplement motion recovery math casually.

## z_int

Current pseudo-label fields include:

- contact body-part state;
- contact target as closest-surface xyz;
- 3-class interaction phase;
- support state;
- object COM/rotation in the generator conditioning frame.

Detailed thresholds and label history live in
`analyses/pseudo_label_pipeline.md`.

## Pipeline

Stage 1: pseudo-label extraction.

- Produces temporal interaction labels from InterAct data.
- Current label track: v11.

Stage A: Interaction Predictor.

- Input: text/object/initial motion context.
- Output: predicted `z_int`.
- Shipped state: v6.
- Predictor of record: server checkpoint `runs/training/predictor/final.pt`;
  local sync may only include eval JSONs.

Stage B: Motion Generator.

- Backbone: MoMask MaskTransformer + ResidualTransformer + VQ-VAE.
- Goal: condition motion generation on text plus `z_int`.
- Current implementation includes:
  - base-token `z_int` adapter;
  - residual-transformer `z_int` conditioning;
  - decoded contact auxiliary loss;
  - full-RVQ decoded-contact path;
  - v15 alignment-aware decoded loss terms for wrong-part margin and
    contact-segment consistency;
  - full-RVQ sampling-time target guidance for offline eval.
- Active bottleneck: v14 improves one-shot contact to `27.37 cm`, but K64
  alignment shows too few samples bind the correct body part to the correct
  object-local patch/timing.

Stage C: Joint Finetune.

- Deferred.
- Intended to align predicted `z_int`, generated motion, and extracted
  interaction signals after Stage B is healthy.

## Metrics

Main Stage B contact metric:

```text
mean_min_dist_per_frame
```

Do not use contact distance alone as the final success criterion. Current
Stage B eval also tracks:

- moving-object temporal coupling;
- contact alignment to GT roundtrip;
- correct GT body-part recall;
- same-part object-local target error.

Use matched sets when comparing:

- GT original;
- GT VQ roundtrip;
- full PIANO generation;
- text_only generation;
- swap/object-mismatched generation.

The old canonical 5-clip set is only a smoke test. Offline conclusions should
use the stratified 80-clip protocol unless explicitly marked otherwise.

## Current Best Numbers

v14 sampled-ST best_contact on matched 80 clips:

| condition | value |
|---|---:|
| GT original | 13.09 cm |
| GT roundtrip | 18.47 cm |
| full | 27.37 cm |
| text_only | 57.82 cm |
| swap | 74.79 cm |

v14 K64 alignment oracle remeasures at `18.71 cm` contact but only `0.2496`
correct GT-part recall and `40.30 cm` same-part local error. v15 is the
implemented next branch to beat this baseline.

## Code Layout

Follow Python src-layout:

- importable package code: `src/piano/`;
- direct runnable scripts: `scripts/`;
- configs: `configs/`;
- local outputs: `runs/`;
- local memory docs: root `*.md` and `analyses/`.

Stable console scripts, if registered in `pyproject.toml`, must point to package
modules under `src/piano/`. Ad-hoc Python entry scripts with `argparse` and
`if __name__ == "__main__"` belong in `scripts/`.

## Important Files

Training:

- `src/piano/training/train_generator.py`
- `src/piano/training/decoded_contact_loss.py`
- `src/piano/training/contact_eval.py`

Models:

- `src/piano/models/motion_generator.py`
- `src/piano/models/backbones/momask_adapter.py`

Evaluation:

- `src/piano/data/eval_sampling.py`
- `scripts/stage_b_generator/`
- `runs/eval/*/summary.json`

## Environment

Server:

```bash
cd /media/gpu-server-1/4TB_for_data/Cai/PIANO/PIANO
conda activate piano
```

Local:

```text
e:\Project\2026-04-13
```

Use the local `piano` conda environment for validation. If a dependency is
missing and clearly belongs in that env, install it there.
