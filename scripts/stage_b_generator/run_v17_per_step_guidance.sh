#!/usr/bin/env bash
# Evaluate Stage B v17 per-step decoded-geometric guidance on an existing
# checkpoint. NO retraining — this branch is purely an inference-time
# addition that runs MaskControl-style per-MaskGIT-iteration logit
# optimisation on top of any v14/v15/v16 ckpt.
#
# See analyses/2026-05-01_per_step_guidance_design.md for the design.
#
# v17-C (default): per-step guidance only, no post-hoc.
#   bash scripts/stage_b_generator/run_v17_per_step_guidance.sh
#
# v17-B (sanity baseline = current v16 full_guided, post-hoc only):
#   PER_STEP_ITERS=0 GUIDANCE_STEPS=30 \
#     bash scripts/stage_b_generator/run_v17_per_step_guidance.sh
#
# v17-D (stacked = canonical MaskControl recipe):
#   GUIDANCE_STEPS=30 PER_STEP_ITERS=10 \
#     bash scripts/stage_b_generator/run_v17_per_step_guidance.sh
#
# v17-E (per-step budget sweep — single override at a time):
#   PER_STEP_ITERS=20 EVAL_PREFIX=stageB_v0_17_per_step_v16bc_iters20 \
#     bash scripts/stage_b_generator/run_v17_per_step_guidance.sh
#
# Source ckpt is v16 best_contact by default. Override with
# SOURCE_RUN_DIR + SOURCE_CFG to point at a different training run
# (e.g. v14 best_contact for a fairer ablation against the v14 K=16
# oracle baseline).

set -euo pipefail

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-runs/training/generator_v16_alignment_mirror}"
SOURCE_CFG="${SOURCE_CFG:-configs/training/generator_v16_alignment_mirror.yaml}"

export TRAIN="${TRAIN:-0}"
export EVAL="${EVAL:-1}"
export DUMP_WANDB="${DUMP_WANDB:-0}"

export CFG="${CFG:-$SOURCE_CFG}"
export RUN_DIR="${RUN_DIR:-$SOURCE_RUN_DIR}"
export RUN_NAME="${RUN_NAME:-stageB_v17_per_step_guidance_eval}"
export EVAL_PREFIX="${EVAL_PREFIX:-stageB_v0_17_per_step_v16bc}"
export CKPTS="${CKPTS:-best_contact}"
export NUM_CLIPS="${NUM_CLIPS:-80}"
export SEED="${SEED:-42}"
export SUMMARY_DETAIL="${SUMMARY_DETAIL:-compact}"

# Per-step decoded-geometric guidance (v17-C default).
export PER_STEP_ITERS="${PER_STEP_ITERS:-10}"
export PER_STEP_LR="${PER_STEP_LR:-6e-2}"
export PER_STEP_TEMPERATURE="${PER_STEP_TEMPERATURE:-1.0}"
export PER_STEP_START_STEP="${PER_STEP_START_STEP:-0}"
# v17-H (B2): part_margin + segment_consistency in per-step inner loss.
# Default 0.0 = back-compat. Sweep: PER_STEP_PART_MARGIN_WEIGHT ∈ {0.5, 1.0, 2.0},
# PER_STEP_SEGMENT_CONSISTENCY_WEIGHT ∈ {0.1, 0.5, 1.0}. See
# analyses/2026-05-01_v17_re_diagnosis.md §B2.
export PER_STEP_PART_MARGIN_WEIGHT="${PER_STEP_PART_MARGIN_WEIGHT:-0.0}"
export PER_STEP_PART_MARGIN_M="${PER_STEP_PART_MARGIN_M:-0.08}"
export PER_STEP_SEGMENT_CONSISTENCY_WEIGHT="${PER_STEP_SEGMENT_CONSISTENCY_WEIGHT:-0.0}"

# Post-hoc final-stage guidance disabled by default in v17-C. Set to 30
# (and optionally GUIDANCE_LAYERS=full_rvq) to stack with per-step.
export GUIDANCE_STEPS="${GUIDANCE_STEPS:-0}"
export GUIDANCE_LAYERS="${GUIDANCE_LAYERS:-full_rvq}"
export GUIDANCE_LOSS="${GUIDANCE_LOSS:-target}"
export GUIDANCE_LR="${GUIDANCE_LR:-6e-2}"
export GUIDANCE_INIT_SCALE="${GUIDANCE_INIT_SCALE:-3.0}"
export GUIDANCE_RESIDUAL_SEED="${GUIDANCE_RESIDUAL_SEED:-42}"

if [[ ! -f "${RUN_DIR}/best_contact.pt" && ! -f "${RUN_DIR}/best_val.pt" && ! -f "${RUN_DIR}/final.pt" ]]; then
  echo "ERROR: no checkpoint found under ${RUN_DIR}" >&2
  echo "  Override SOURCE_RUN_DIR= to point at a training dir with .pt files." >&2
  exit 1
fi

echo "============================================================"
echo "v17 per-step guidance eval"
echo "  source ckpt dir: ${RUN_DIR}"
echo "  ckpts: ${CKPTS}"
echo "  per_step_iters=${PER_STEP_ITERS} lr=${PER_STEP_LR} T=${PER_STEP_TEMPERATURE} start=${PER_STEP_START_STEP}"
echo "  per_step_part_margin_weight=${PER_STEP_PART_MARGIN_WEIGHT} m=${PER_STEP_PART_MARGIN_M} segment_consistency=${PER_STEP_SEGMENT_CONSISTENCY_WEIGHT}"
echo "  post_hoc guidance_steps=${GUIDANCE_STEPS} layers=${GUIDANCE_LAYERS} loss=${GUIDANCE_LOSS}"
echo "  output prefix: ${EVAL_PREFIX}"
echo "============================================================"

exec bash scripts/stage_b_generator/run_v13_target_trajectory.sh
