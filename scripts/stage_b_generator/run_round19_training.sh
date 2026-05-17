#!/usr/bin/env bash
# Round-19 paired training — Plan A vs S1-O, 6 seeds × 2 modes = 12 runs.
#
# Strategy: per-seed concurrent pair across the two A6000s.
#   GPU0 = Plan A (configs/training/coarse_prior_s1a_cmc.yaml)
#   GPU1 = S1-O   (configs/training/coarse_prior_s1o_root0_world.yaml)
# Both launch in background, script waits for both before advancing to
# the next seed. This keeps each paired (seed, mode) couple on similar
# wallclock + hardware state, which the variance protocol relies on.
#
# Seeds: 42 43 44 45 46 47 (matches PLAN.md §"Immediate priority — Round 19").
# Per-run wallclock: ~5-6h (100k steps × bf16 × bs 16 × accum 4 on A6000).
# Total wallclock: ~30-36h with the two GPUs running in parallel.
#
# Per-run stdout/stderr → runs/training/round19_launch_logs/<run_name>.log
# Per-run output_dir (ckpt + loss_log.json + wandb metadata) is set
# explicitly via --output-dir so the YAML default doesn't get clobbered
# across seeds.
#
# Usage (from anywhere; script cd's to repo root):
#   bash scripts/stage_b_generator/run_round19_training.sh
#
# Resume / partial recovery: if interrupted, re-running this script will
# RESTART each in-progress seed from scratch (trainer has no resume).
# To skip already-completed seeds, comment them out of the SEEDS array.

set -euo pipefail
cd "$(dirname "$0")/../.."

SEEDS=(42 43 44 45 46 47)
LOG_DIR="runs/training/round19_launch_logs"
mkdir -p "${LOG_DIR}"

CONFIG_PLAN_A="configs/training/coarse_prior_s1a_cmc.yaml"
CONFIG_S1O="configs/training/coarse_prior_s1o_root0_world.yaml"

echo "[round19-train] repo root: $(pwd)"
echo "[round19-train] log dir:   ${LOG_DIR}"
echo "[round19-train] python:    $(which python)"
echo "[round19-train] seeds:     ${SEEDS[*]}"
echo "[round19-train] Plan A on GPU0, S1-O on GPU1, paired concurrent per seed"
echo

for SEED in "${SEEDS[@]}"; do
    RUN_PLAN_A="stage1_s1a_cmc_round19_seed${SEED}"
    RUN_S1O="stage1_s1o_round19_seed${SEED}"
    OUT_PLAN_A="runs/training/${RUN_PLAN_A}"
    OUT_S1O="runs/training/${RUN_S1O}"
    LOG_PLAN_A="${LOG_DIR}/${RUN_PLAN_A}.log"
    LOG_S1O="${LOG_DIR}/${RUN_S1O}.log"

    echo "===== seed=${SEED} : launching paired Plan A (GPU0) + S1-O (GPU1) ====="
    echo "  Plan A: ${OUT_PLAN_A}  log=${LOG_PLAN_A}"
    echo "  S1-O  : ${OUT_S1O}     log=${LOG_S1O}"

    CUDA_VISIBLE_DEVICES=0 \
    python src/piano/training/train_coarse_prior.py \
        --config "${CONFIG_PLAN_A}" \
        --seed "${SEED}" \
        --output-dir "${OUT_PLAN_A}" \
        --checkpoint-name final.pt \
        > "${LOG_PLAN_A}" 2>&1 &
    PID_PLAN_A=$!

    CUDA_VISIBLE_DEVICES=1 \
    python src/piano/training/train_coarse_prior.py \
        --config "${CONFIG_S1O}" \
        --seed "${SEED}" \
        --output-dir "${OUT_S1O}" \
        --checkpoint-name final.pt \
        > "${LOG_S1O}" 2>&1 &
    PID_S1O=$!

    echo "  PIDs: Plan A=${PID_PLAN_A}  S1-O=${PID_S1O}"
    echo "  waiting for both to finish before next seed..."

    # Wait on each; capture exit codes individually so one failure
    # doesn't lose information about the other.
    set +e
    wait ${PID_PLAN_A}; RC_PLAN_A=$?
    wait ${PID_S1O};    RC_S1O=$?
    set -e

    echo "  exit codes: Plan A=${RC_PLAN_A}  S1-O=${RC_S1O}"
    if [[ ${RC_PLAN_A} -ne 0 || ${RC_S1O} -ne 0 ]]; then
        echo "[round19-train] FAIL at seed=${SEED} — see logs:"
        echo "  ${LOG_PLAN_A}"
        echo "  ${LOG_S1O}"
        exit 1
    fi
    echo "  seed=${SEED} pair complete ✓"
    echo
done

echo "===== Packing all training logs + outputs metadata ====="
tar czf "runs/training/round19_training.tar.gz" \
    "${LOG_DIR}/" \
    runs/training/stage1_s1a_cmc_round19_seed*/loss_log.json \
    runs/training/stage1_s1o_round19_seed*/loss_log.json \
    runs/training/stage1_s1a_cmc_round19_seed*/config.yaml \
    runs/training/stage1_s1o_round19_seed*/config.yaml \
    2>/dev/null || true

echo
echo "[round19-train] ALL 12 RUNS COMPLETE"
echo "[round19-train] upload: runs/training/round19_training.tar.gz"
echo "[round19-train] (full ckpts stay on server; metadata bundle is small)"
