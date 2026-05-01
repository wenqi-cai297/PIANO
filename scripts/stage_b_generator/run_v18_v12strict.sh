#!/usr/bin/env bash
# Stage B v18 — train on v12 strict pseudo-labels.
#
# Prereq: server-side v12 extraction must already have run:
#   bash scripts/stage1_pseudo_labels/extract_v12_strict_interact.sh
# This produces <subset>/pseudo_labels/v12_strict/<seq>.npz files.
#
# v18 reuses v16's loss / architecture / augmentation; the ONLY change
# is that pseudo_label_subdir = "pseudo_labels/v12_strict". This isolates
# the contribution of the pseudo-label definition change.
#
# Predicted outcomes (analyses/2026-05-03_pseudo_label_v12_strict_design.md):
#   raw correct_part_recall:    0.176 (v16) → 0.30+
#   guided correct_part_recall: 0.292 (v17-E.50) → 0.40+
#   visual: real contact instead of "approach"
#
# Usage:
#   bash scripts/stage_b_generator/run_v18_v12strict.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

export TRAIN="${TRAIN:-1}"
export EVAL="${EVAL:-1}"
export DUMP_WANDB="${DUMP_WANDB:-1}"

export CFG="${CFG:-configs/training/generator_v18_v12strict.yaml}"
export RUN_DIR="${RUN_DIR:-runs/training/generator_v18_v12strict}"
export RUN_NAME="${RUN_NAME:-stageB_v18_v12strict}"
export EVAL_PREFIX="${EVAL_PREFIX:-stageB_v0_18_v12strict}"
export CKPTS="${CKPTS:-best_contact best_val final}"
export NUM_CLIPS="${NUM_CLIPS:-80}"
export SEED="${SEED:-42}"
export SUMMARY_DETAIL="${SUMMARY_DETAIL:-compact}"

# Per-step inference is the recommended ship config — same as v17-E.20.
# Once v18 ckpt is trained, re-evaluate with this inference recipe to
# compare apples-to-apples against v17-E.20 + v16 final.pt.
export PER_STEP_ITERS="${PER_STEP_ITERS:-20}"
export PER_STEP_LR="${PER_STEP_LR:-6e-2}"
export PER_STEP_TEMPERATURE="${PER_STEP_TEMPERATURE:-1.0}"
export PER_STEP_START_STEP="${PER_STEP_START_STEP:-0}"
export PER_STEP_GUMBEL_SCALE="${PER_STEP_GUMBEL_SCALE:-0.0}"
export GUIDANCE_STEPS="${GUIDANCE_STEPS:-0}"
export GUIDANCE_LAYERS="${GUIDANCE_LAYERS:-full_rvq}"
export GUIDANCE_LOSS="${GUIDANCE_LOSS:-target}"
export GUIDANCE_LR="${GUIDANCE_LR:-6e-2}"
export GUIDANCE_INIT_SCALE="${GUIDANCE_INIT_SCALE:-3.0}"
export GUIDANCE_RESIDUAL_SEED="${GUIDANCE_RESIDUAL_SEED:-42}"

echo "============================================================"
echo "v18 v12_strict train + eval"
echo "  config: ${CFG}"
echo "  output: ${RUN_DIR}"
echo "  pseudo_label_subdir: pseudo_labels/v12_strict"
echo "  ckpts: ${CKPTS}"
echo "============================================================"

exec bash scripts/stage_b_generator/run_v13_target_trajectory.sh
