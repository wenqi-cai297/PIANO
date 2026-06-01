# PIANO

**Physically-Informed Adaptive iNteraction Orchestration** — object-adaptive
human motion generation via structured interaction latents.

PIANO generates physically plausible full-body human motion that interacts
with an arbitrary 3D object (chair, box, suitcase, …) from a text
description and a 3D point cloud of the object. The system is trained on the
InterAct subset of human–object interaction datasets (CHAIRS, IMHD,
NeuralDome, OMOMO).

The core challenge — generating motion that respects the object's geometry
(hand on the chair's seat, not floating 10 cm away) without overfitting to
any one pose mode — is attacked by **cascading three diffusion models**
through structured intermediate representations rather than a single
end-to-end network.

## Pipeline overview

```
text + object pointcloud
        │
        ▼
   ┌────────────┐    ┌────────────┐    ┌──────────────┐
   │  Stage-1   │───▶│ Stage-1.5  │───▶│   Stage-2    │───▶ 135-D SMPL motion
   │   23-D     │    │  C41 + S4  │    │  PB1 / Pose  │     (rot6d×22 + root)
   │ coarse plan│    │  31-D scaf │    │   Body       │
   └────────────┘    └────────────┘    └──────────────┘
```

| Stage | Output | What it carries | Channels |
|---|---|---|---:|
| **Stage-1** (`stage1_coarse`) | coarse trajectory + orientation plan | root_local xyz, root velocity xyz, yaw sin/cos + ang_vel, pelvis rot6d, spine3 rot6d, head/shoulder heights | **23** |
| **Stage-1.5** (C41 + S4) | interaction scaffold | C41 = 18-D pelvis-local Δxyz for wrist/knee/neck/pelvis joints; S4 = 13-D foot stance / gait phase / footstep markers | **31** |
| **Stage-2 PB1** | full motion | 22 joint rot6d + root world position | **135** |

Each stage is a DDPM (cosine schedule, 1000 steps, x0-prediction) on a
DiT-Zero + PixArt cross-attention backbone. Stages downstream of Stage-1
inject the upstream cond via PB1's AdaLN-S4 routing pattern (zero-init
injection so step-0 behavior equals the no-cond baseline; the cond
contribution grows only if it helps).

All three stages share infrastructure under `src/piano/`:

- `piano.data.stage1_coarse_oracle` — extracts the GT 23-D plan from raw
  motion (batched torch implementation, equivalence-tested against the
  reference numpy extractor).
- `piano.data.stage2_oracle_conditions` — extracts C41 + S4 conds.
- `piano.models.{stage1_trajectory, stage1p5_interaction, motion_anchordiff}`
  — the three denoisers.
- `piano.training.{train_stage1, train_stage1p5, train_anchordiff}` — the
  three trainers (HuggingFace Accelerate, bf16, multi-GPU).
- `piano.inference.sample_substitute_conds` — samples each stage's output
  into a per-clip `.npz` cache, used as the substitute cond for the next
  stage's training/diag.

## Repository layout

```
src/piano/                         importable library code (Python src layout)
  data/                            datasets, oracle cond extractors, FK
  models/                          DiT denoisers, object encoder, backbones
  training/                        per-stage trainers + loss helpers
  inference/                       sampling, substitute-conds cache, viz
  evaluation/, sampling/, checks/

scripts/                           runnable entry points (run directly)
  prep/                            data bring-up
  stage1_pseudo_labels/            pseudo-label extraction + QA
  stage_a_predictor/               Stage-1 / Stage-1.5 ablation matrices
  stage_a_generator/               trainer launchers, sync packers, diags
  stage_b_generator/               Stage-2 / PB1 training + diagnostics
  eval/, vis/, checks/

configs/training/                  one YAML per ablation cell
analyses/                          dated session docs (R<N>_<slug>.md)
tests/                             pytest suite (currently 121+ stage-loss
                                   tests, plus stage1/stage1p5/PB1 model + 
                                   ablation manifest tests)

PROGRESS.md                        round-by-round status (most recent first)
PLAN.md                            next-step decision tree
ANALYSIS.md                        cross-round analytical notes
```

## Setup

Python 3.10+. Install in editable mode:

```bash
pip install -e .
# optional extras
pip install -e ".[dev,viz,wandb]"
```

Hardware: training is calibrated for **3× RTX 5080 (16 GB)** on the
internal `5080x3` host; the cascade is small enough that 1× 5080 also fits
(adjust `batch_size` and accumulation).

Dataset path:

```bash
export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
```

The processed dataset includes the CHAIRS, IMHD, NeuralDome, and
OMOMO-correct-v2 subsets with frame-level pseudo labels for contact, gait
phase, and foot stance.

## Running

The active training entry point registered in `pyproject.toml` is the
Stage-2 trainer (`piano-train-anchordiff`); Stage-1 and Stage-1.5 are
invoked directly:

```bash
# Stage-2 PB1 (the shipped reference)
piano-train-anchordiff --config configs/training/anchordiff_r29_pb_a1_adaln_s4.yaml

# Stage-1 / Stage-1.5 (direct script invocation)
python -u src/piano/training/train_stage1.py   --config configs/training/stage1_<variant>.yaml
python -u src/piano/training/train_stage1p5.py --config configs/training/stage1p5_<variant>.yaml
```

Every ablation round is wrapped in a launcher that handles
config-generation, train, downstream-diag, summary, and sync-pack:

```bash
bash scripts/stage_a_generator/run_round<N>_<round>.sh           # full matrix
bash scripts/stage_a_generator/run_round<N>_<round>.sh --dry-run # show commands
bash scripts/stage_a_generator/pack_round<N>_<round>_sync.sh     # tarball back
```

Tests:

```bash
pytest -q tests/                                    # full suite
pytest -q tests/test_stage1_losses.py tests/test_stage1p5_losses.py
```

## Visualisation

```bash
piano-visualize-motion --input runs/.../sample.npy --object path/to/obj.obj
```

Note: outputs from `qual_eval` / `gt_vq_roundtrip` save **raw-scale**
motion_263 — do **not** pass `--mean`/`--std` to `visualize_motion` on those
files (double-denorm bug).

## Development status

PIANO has been iterated through ~40 ablation rounds. The headline story is
"split the monolithic motion generator into a 3-stage cascade, then fix the
mode collapse that ended up compressed into Stage-1." Each row below is one
ablation round; verdicts come from the sustained-contact / gait /
body-action / soft-stance diag suite on the same 48 balanced val clips.

The full chronology is in [`PROGRESS.md`](PROGRESS.md) (most recent first)
and the next-step decision tree is in [`PLAN.md`](PLAN.md).

### Stage-2 — Pose-Body diffusion (R23–R29)

| Round | Topic | Verdict |
|---|---|---|
| R23 | Motion-faithful + visual + temporal diag baseline | Established the 48-clip ship-gate metric set |
| R26 | Tier-0 oracle hint vs Tier-1 temporal losses | Tier-0 wins contact; Tier-1 wins gait but degrades contact |
| R27 | T0/T1 sequencing | T0 first, T1 only after contact stabilises |
| R28 | Oracle-interface refinement (A0 input_add) | A0 wins both contact + gait at 48-clip overfit |
| R29 | Loss-strategy + cond-injection ablation | **PB1 `r29_pb_a1_adaln_s4` SHIPPED** — val drift_max **7.55 cm**, locked as Stage-2 reference |

PB1's cond contract is fixed: stage1_coarse (23-D z-scored) + C41 (18-D
raw) + S4 (13-D raw). Anything upstream must match this exactly.

### Stage-1.5 — Interaction-Plan diffusion (R32–R38)

| Round | Topic | Verdict |
|---|---|---|
| R32 V7 | 6-variant anti-bug loss matrix (B1–B5 fixes) | CLOSED **NEGATIVE** — all variants close < 1 cm wrist drift; per-channel loss design exhausted |
| R32 Phase 1 audit | Failure-mode characterisation | NOT mode-collapsed; 5 localised bugs identified |
| R33 V0 | Per-block obj_xattn (DiT-XL pattern) — every Stage-1.5 DiT layer | Structural fix; closed wrist drift gap further |
| R34 V2-A | r34_wrist_lowband λ=0.005 substrate | drift_max **13.86 cm** (oracle Stage-1 cond), best Stage-1.5 ckpt pre-R38 |
| R36 | Temporal dynamics losses | scale-dominate failure; loss weight too high |
| R37 | C41 frame-level dynamics losses | **CATASTROPHIC** (drift_max 86–96 cm) — C41 pelvis-local Δxyz dynamics supervision pushes pred into a sub-space PB1 cannot consume |
| R38 | init_pose F1 + contact-window wrist MSE (4-cell B0/B1/B2/B3) | **B1 SHIPPED** — drift_max **11.89 cm** oracle, 36.30 cm generated Stage-1 cond |

R38 shipped B1 (`stage1p5_r38_b1_init_pose`) as the current Stage-1.5
reference. The 36.30 cm generated-cond number is what motivated R40 below.

### Stage-1 — Coarse-Plan diffusion (R31, R35, R40)

| Round | Topic | Verdict |
|---|---|---|
| R31 V0 | Baseline trajectory predictor | val drift_max ~18.5 cm; +11 cm regression vs PB1 oracle reference |
| R31 V2 | 6-variant rot6d + height-FK loss ablation | CLOSED **NEGATIVE** — all 6 within 3 % of each other; loss-design hypothesis refuted |
| R31 V7 | 6-variant anti-collapse stack (channel moment match + yaw aggregate + cm-space FK) | drift_max 17.52 cm; small gain |
| R31 V8 | 7-variant wrist FK + frame-0 anchor (init_pose F1/F2) | V6 (W2 + F1) shipped as Stage-1 mainline; ~1.5 cm wrist drift improvement |
| R35 | Stage-1 generated-cond OOD audit on V8 V6 | Confirmed mode collapse: `velocity_xzy` vel_ratio **0.379**, `pelvis_rot6d` std_ratio **0.493**, vel_ratio **0.474** vs GT |
| **R40** | **Plan-sampler ablation (4-cell C0/C1/C2/C3)** | **IMPLEMENTED + PUSHED** (`c0421e6`, 2026-06-01). Channel-weighted MSE + plan-invariant loss. Server training in progress. |

### Active long runs

- **R40 Stage-1 plan-sampler matrix** on `5080x3` (0,2) — 4 cells × ~10 min
  train + direct-PB1 diag + plan-quality diag + optional full cascade +
  K-sample diversity. C0 baseline confirmed bit-identical to V8 V6 in
  initial epochs.

### Reference checkpoints

| Role | Path | Frozen since |
|---|---|---|
| Stage-2 / PB1 ship | `runs/training/stageB_anchordiff_r29_pb_a1_adaln_s4/final.pt` | 2026-05-29 |
| Stage-1.5 ship | `runs/training/stage1p5_r38_b1_init_pose/final.pt` | 2026-06-01 |
| Stage-1 mainline | `runs/training/stage1_v8_v6_full_f1/final.pt` (under R40 evaluation) | 2026-05-30 |

### Operating principles (learned the hard way)

These show up across the round history above and are now project
conventions:

- **Inference-feasibility audit before any cond addition.** R38 ruled out
  `contact_target_xyz` / `contact_state` as Stage-1.5 conds because both
  derive from GT motion and aren't available end-to-end.
- **`val_mse_*` is not the ship metric.** R37 and R40 both have variants
  where val MSE goes down while downstream drift gets worse.
  Ship cell is selected on direct downstream drift + plan-quality audit on
  `final.pt`, not on `best_val.pt`.
- **No GT-derived cond, no architecture scaling, no untested loss weight
  ≥ 3× the baseline term.** The R36 scale-dominate disaster (loss weight
  too high) and the R37 dynamics-on-C41 disaster (entangled supervision)
  are the two failure modes the calibration audit guards against.
- **Two-axis ship gate.** Each stage transition evaluates on
  (a) relative-to-prior and (b) absolute-functional metrics before passing
  to the next stage. A regression on the absolute axis blocks ship even if
  the relative axis improves.

See [`PROGRESS.md`](PROGRESS.md), [`PLAN.md`](PLAN.md), and the dated
`analyses/YYYY-MM-DD_*.md` docs for the full record per round.
