#!/usr/bin/env bash
# Round-20 paired training: Plan A vs S1-O with the corrected training recipe.
#
# Strategy changes vs Round-19:
#   - 40k total steps instead of 100k.
#   - save cadence aligned with validation cadence (5k/5k).
#   - x0-pred Min-SNR-gamma weighting enabled in the Stage-1 trainer.
#   - deterministic mirror doubling enabled in the Stage-1 cache dataset.
#
# Method-iteration mode: single seed 42, single GPU (cuda:0), Plan A then S1-O serial.
# Multi-seed stability sweep is deferred until the method is locked.
#
# CUDA0 = Plan A (configs/training/coarse_prior_s1a_cmc.yaml), then
# CUDA0 = S1-O  (configs/training/coarse_prior_s1o_root0_world.yaml)
#
# Prereq: refresh CLIP text cache with direction-mirrored text variants,
# then run the full preflight gate:
#   bash scripts/stage_b_generator/run_round20_recache_text.sh
#   python scripts/stage_b_generator/test_stage1_round18_preflight.py
#
# Usage:
#   bash scripts/stage_b_generator/run_round20_training.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

SEEDS=(42)
LOG_DIR="runs/training/round20_launch_logs"
mkdir -p "${LOG_DIR}"

CONFIG_PLAN_A="configs/training/coarse_prior_s1a_cmc.yaml"
CONFIG_S1O="configs/training/coarse_prior_s1o_root0_world.yaml"

echo "[round20-train] repo root: $(pwd)"
echo "[round20-train] log dir:   ${LOG_DIR}"
echo "[round20-train] python:    $(which python)"
echo "[round20-train] seeds:     ${SEEDS[*]}"
echo "[round20-train] mode:      Plan A then S1-O, serial on CUDA0 (method-iteration)"
echo

for SEED in "${SEEDS[@]}"; do
    RUN_PLAN_A="stage1_s1a_cmc_round20_seed${SEED}"
    RUN_S1O="stage1_s1o_round20_seed${SEED}"
    OUT_PLAN_A="runs/training/${RUN_PLAN_A}"
    OUT_S1O="runs/training/${RUN_S1O}"
    LOG_PLAN_A="${LOG_DIR}/${RUN_PLAN_A}.log"
    LOG_S1O="${LOG_DIR}/${RUN_S1O}.log"

    echo "===== seed=${SEED}: Plan A on CUDA0 ====="
    echo "  Plan A: ${OUT_PLAN_A}  log=${LOG_PLAN_A}"

    set +e
    CUDA_VISIBLE_DEVICES=0 \
    python src/piano/training/train_coarse_prior.py \
        --config "${CONFIG_PLAN_A}" \
        --seed "${SEED}" \
        --output-dir "${OUT_PLAN_A}" \
        --checkpoint-name final.pt \
        > "${LOG_PLAN_A}" 2>&1
    RC_PLAN_A=$?
    set -e

    echo "  exit code: Plan A=${RC_PLAN_A}"
    if [[ ${RC_PLAN_A} -ne 0 ]]; then
        echo "[round20-train] FAIL at seed=${SEED} Plan A; see log:"
        echo "  ${LOG_PLAN_A}"
        exit 1
    fi

    echo "===== seed=${SEED}: S1-O on CUDA0 ====="
    echo "  S1-O  : ${OUT_S1O}     log=${LOG_S1O}"

    set +e
    CUDA_VISIBLE_DEVICES=0 \
    python src/piano/training/train_coarse_prior.py \
        --config "${CONFIG_S1O}" \
        --seed "${SEED}" \
        --output-dir "${OUT_S1O}" \
        --checkpoint-name final.pt \
        > "${LOG_S1O}" 2>&1
    RC_S1O=$?
    set -e

    echo "  exit code: S1-O=${RC_S1O}"
    if [[ ${RC_S1O} -ne 0 ]]; then
        echo "[round20-train] FAIL at seed=${SEED} S1-O; see log:"
        echo "  ${LOG_S1O}"
        exit 1
    fi
    echo "  seed=${SEED} pair complete (Plan A + S1-O)"
    echo
done

echo "===== Packing all training logs + outputs metadata ====="
tar czf "runs/training/round20_training.tar.gz" \
    "${LOG_DIR}/" \
    runs/training/stage1_s1a_cmc_round20_seed*/loss_log.json \
    runs/training/stage1_s1o_round20_seed*/loss_log.json \
    runs/training/stage1_s1a_cmc_round20_seed*/training_summary.json \
    runs/training/stage1_s1o_round20_seed*/training_summary.json \
    runs/training/stage1_s1a_cmc_round20_seed*/config.yaml \
    runs/training/stage1_s1o_round20_seed*/config.yaml \
    2>/dev/null || true

echo
echo "[round20-train] ALL RUNS COMPLETE (seed=${SEEDS[*]}: Plan A + S1-O)"
echo "[round20-train] upload: runs/training/round20_training.tar.gz"
echo "[round20-train] full ckpts stay on the training machine"
