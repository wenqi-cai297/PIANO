#!/usr/bin/env bash
# Round-25 P0 full diagnostic bundle launcher — DUAL-GPU parallel.
#
# Server has 2× A6000 (cuda:0 + cuda:1). The bundle uses 3 parallel
# phases interleaved with 2 sequential prep stages:
#
#   PREP-A   D1 propose val + train (CPU + minimal GPU, sequential)
#   PREP-B   D1 curate + D4 indices (CPU only, sequential)
#   PHASE-A  D2 (cuda:0) || D3 (cuda:1)         — diversity + sampler
#   PHASE-B  D4-8 (cuda:0) || D4-16 (cuda:1)    — overfit
#   PHASE-C  D5-V0 (cuda:0) || D5-V1 (cuda:1)   — loss-weight smoke
#   PHASE-D  D5-V2 (cuda:0)                     — remaining V2 solo
#
# Estimated wall (2× A6000):
#   PREP   ~5 min
#   PHASE-A ~20-30 min (max of D2 ~25, D3 ~15)
#   PHASE-B ~60 min   (max of D4-8 ~30, D4-16 ~60)
#   PHASE-C ~30 min
#   PHASE-D ~30 min
#   TOTAL  ~2.5 hours (vs 4-6h sequential)
#
# Prerequisites:
#   1. git pull
#   2. bash scripts/stage_b_generator/run_round25_make_local_configs.sh
#   3. v26 ckpt + Stage-1 ckpt + Stage-1 cache + Stage-2 cache present
#
# Usage:
#   bash scripts/stage_b_generator/run_round25_full_diagnostic.sh
#
# Override to single-GPU mode (only cuda:0, all stages sequential):
#   ROUND25_SINGLE_GPU=1 bash <this script>

set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="runs/round25_diagnostic"
mkdir -p "${LOG_DIR}"

V26_CFG_LOCAL="configs/training/anchordiff_v26_FULL_DATA_local.yaml"
V26_CKPT="runs/training/stageB_anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA/final.pt"
S1_CKPT="runs/training/stage1_s1o_round20_seed42/final.pt"
S1_CACHE="cache/stage1_coarse_v1_objtraj_root0_world_round18_fix"

SINGLE_GPU="${ROUND25_SINGLE_GPU:-0}"

# ---------- preflight ----------
for F in "${V26_CFG_LOCAL}" "${V26_CKPT}" "${S1_CKPT}"; do
    if [[ ! -e "${F}" ]]; then
        echo "ERROR: missing prerequisite: ${F}"
        echo "Did you run run_round25_make_local_configs.sh and verify ckpt paths?"
        exit 1
    fi
done
if [[ ! -d "${S1_CACHE}" ]]; then
    echo "ERROR: missing Stage-1 cache: ${S1_CACHE}"
    exit 1
fi

# Foreground step with tee to log.
run_step() {
    local NAME="$1"; shift
    local LOG="${LOG_DIR}/${NAME}.log"
    local T0
    T0=$(date +%s)
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] START ${NAME}"
    echo "    log: ${LOG}"
    echo "================================================================"
    PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$@" 2>&1 | tee "${LOG}"
    local T1
    T1=$(date +%s)
    echo "[$(date '+%F %T')] DONE ${NAME} in $((T1-T0))s"
}

# Background step (caller backgrounds with &). Writes ONLY to log,
# not stdout, so concurrent runs don't interleave terminal output.
run_step_bg() {
    local NAME="$1"; shift
    local GPU="$1"; shift
    local LOG="${LOG_DIR}/${NAME}.log"
    local T0
    T0=$(date +%s)
    {
        echo "================================================================"
        echo "[$(date '+%F %T')] BG-START ${NAME} on cuda:${GPU}"
        echo "    pid: $$  log: ${LOG}"
        echo "================================================================"
        CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$@" 2>&1
        local T1
        T1=$(date +%s)
        echo "[$(date '+%F %T')] BG-DONE ${NAME} in $((T1-T0))s"
    } > "${LOG}" 2>&1
}

# ============================================================
# PREP-A: D1 propose val + train (sequential)
# ============================================================
run_step "d1_propose_val" \
    conda run --no-capture-output -n piano python -u \
        scripts/stage_b_generator/round25_d1_propose_multimodal_candidates.py \
        --config "${V26_CFG_LOCAL}" \
        --output analyses/round25_multimodal_candidates_val.json \
        --bucket val --top-k 250 --min-confidence 0.5

run_step "d1_propose_train" \
    conda run --no-capture-output -n piano python -u \
        scripts/stage_b_generator/round25_d1_propose_multimodal_candidates.py \
        --config "${V26_CFG_LOCAL}" \
        --output analyses/round25_multimodal_candidates_train.json \
        --bucket train --top-k 400 --min-confidence 0.5

# ============================================================
# PREP-B: D1 curate + D4 indices (sequential, fast)
# ============================================================
run_step "d1_curate" \
    conda run --no-capture-output -n piano python -u \
        scripts/stage_b_generator/round25_curate_subsets.py

run_step "d4_build_indices_8" \
    conda run --no-capture-output -n piano python -u \
        scripts/stage_b_generator/round25_d4_build_subset_indices.py \
        --config "${V26_CFG_LOCAL}" \
        --train-selection-json analyses/round25_d4_train_selection.json \
        --n-clips 8 --output analyses/round25_d4_indices_8.json

run_step "d4_build_indices_16" \
    conda run --no-capture-output -n piano python -u \
        scripts/stage_b_generator/round25_d4_build_subset_indices.py \
        --config "${V26_CFG_LOCAL}" \
        --train-selection-json analyses/round25_d4_train_selection.json \
        --n-clips 16 --output analyses/round25_d4_indices_16.json

if [[ "${SINGLE_GPU}" == "1" ]]; then
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] SINGLE-GPU MODE: running D2 → D3 → D4-8 → D4-16 → D5-V0 → D5-V1 → D5-V2 sequentially on cuda:0"
    echo "================================================================"

    run_step "d2_diversity" \
        conda run --no-capture-output -n piano python -u \
            scripts/stage_b_generator/round25_d2_diversity_diagnostic.py \
            --config "${V26_CFG_LOCAL}" --ckpt "${V26_CKPT}" \
            --selection-json analyses/round25_multimodal_eval_subset.json \
            --bucket val --n-samples 8 --cfg-scale 1.0 \
            --output analyses/round25_d2_diversity_stats.json

    run_step "d3_oracle_vs_sampled" \
        conda run --no-capture-output -n piano python -u \
            scripts/stage_b_generator/round25_d3_oracle_vs_sampled.py \
            --config "${V26_CFG_LOCAL}" --ckpt "${V26_CKPT}" \
            --stage1-ckpt "${S1_CKPT}" --stage1-cache-root "${S1_CACHE}" \
            --selection-json analyses/round25_multimodal_eval_subset.json \
            --bucket val --cfg-scale 1.0 --cfg-scale-stage1 1.0 --seed 42 \
            --output analyses/round25_d3_oracle_vs_sampled.json

    for CFG in d4_overfit8 d4_overfit16 \
               d5_v0_baseline d5_v1_hand2x_foot2x d5_v2_hand5x_foot5x; do
        run_step "${CFG}" \
            conda run --no-capture-output -n piano accelerate launch \
                --num_processes 1 --mixed_precision bf16 \
                src/piano/training/train_anchordiff.py \
                --config "configs/training/anchordiff_v26_${CFG}_local.yaml"
    done

else
    # ============================================================
    # DUAL-GPU MODE: 4 parallel phases
    # ============================================================

    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE-A: D2 (cuda:0) || D3 (cuda:1)"
    echo "================================================================"
    run_step_bg "d2_diversity" 0 \
        conda run --no-capture-output -n piano python -u \
            scripts/stage_b_generator/round25_d2_diversity_diagnostic.py \
            --config "${V26_CFG_LOCAL}" --ckpt "${V26_CKPT}" \
            --selection-json analyses/round25_multimodal_eval_subset.json \
            --bucket val --n-samples 8 --cfg-scale 1.0 \
            --output analyses/round25_d2_diversity_stats.json &
    PID_D2=$!

    run_step_bg "d3_oracle_vs_sampled" 1 \
        conda run --no-capture-output -n piano python -u \
            scripts/stage_b_generator/round25_d3_oracle_vs_sampled.py \
            --config "${V26_CFG_LOCAL}" --ckpt "${V26_CKPT}" \
            --stage1-ckpt "${S1_CKPT}" --stage1-cache-root "${S1_CACHE}" \
            --selection-json analyses/round25_multimodal_eval_subset.json \
            --bucket val --cfg-scale 1.0 --cfg-scale-stage1 1.0 --seed 42 \
            --output analyses/round25_d3_oracle_vs_sampled.json &
    PID_D3=$!

    echo "    PID D2=${PID_D2}  D3=${PID_D3}"
    echo "    follow logs: tail -f ${LOG_DIR}/d2_diversity.log ${LOG_DIR}/d3_oracle_vs_sampled.log"
    wait $PID_D2 $PID_D3
    echo "[$(date '+%F %T')] PHASE-A DONE"

    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE-B: D4-8 (cuda:0) || D4-16 (cuda:1)"
    echo "================================================================"
    run_step_bg "d4_overfit8" 0 \
        conda run --no-capture-output -n piano accelerate launch \
            --num_processes 1 --mixed_precision bf16 \
            src/piano/training/train_anchordiff.py \
            --config configs/training/anchordiff_v26_d4_overfit8_local.yaml &
    PID_D4_8=$!

    run_step_bg "d4_overfit16" 1 \
        conda run --no-capture-output -n piano accelerate launch \
            --num_processes 1 --mixed_precision bf16 \
            src/piano/training/train_anchordiff.py \
            --config configs/training/anchordiff_v26_d4_overfit16_local.yaml &
    PID_D4_16=$!

    echo "    PID D4-8=${PID_D4_8}  D4-16=${PID_D4_16}"
    wait $PID_D4_8 $PID_D4_16
    echo "[$(date '+%F %T')] PHASE-B DONE"

    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE-C: D5-V0 (cuda:0) || D5-V1 (cuda:1)"
    echo "================================================================"
    run_step_bg "d5_v0_baseline" 0 \
        conda run --no-capture-output -n piano accelerate launch \
            --num_processes 1 --mixed_precision bf16 \
            src/piano/training/train_anchordiff.py \
            --config configs/training/anchordiff_v26_d5_v0_baseline_local.yaml &
    PID_V0=$!

    run_step_bg "d5_v1_hand2x_foot2x" 1 \
        conda run --no-capture-output -n piano accelerate launch \
            --num_processes 1 --mixed_precision bf16 \
            src/piano/training/train_anchordiff.py \
            --config configs/training/anchordiff_v26_d5_v1_hand2x_foot2x_local.yaml &
    PID_V1=$!

    echo "    PID V0=${PID_V0}  V1=${PID_V1}"
    wait $PID_V0 $PID_V1
    echo "[$(date '+%F %T')] PHASE-C DONE"

    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE-D: D5-V2 (cuda:0) solo"
    echo "================================================================"
    # Could be split V2 (cuda:0) || (next thing on cuda:1) but there's
    # no D5-V3 — V2 runs solo. Foreground tee here since nothing else
    # is on cuda:1 to occupy the terminal.
    run_step "d5_v2_hand5x_foot5x" \
        conda run --no-capture-output -n piano accelerate launch \
            --num_processes 1 --mixed_precision bf16 \
            src/piano/training/train_anchordiff.py \
            --config configs/training/anchordiff_v26_d5_v2_hand5x_foot5x_local.yaml
fi

echo
echo "================================================================"
echo "Round-25 full diagnostic bundle complete."
echo "Outputs:"
echo "  D2 stats:      analyses/round25_d2_diversity_stats.{json,md}"
echo "  D3 stats:      analyses/round25_d3_oracle_vs_sampled.{json,md}"
echo "  D4 runs:       runs/training/stageB_anchordiff_v26_d4_overfit{8,16}/"
echo "  D5 runs:       runs/training/stageB_anchordiff_v26_d5_{v0,v1,v2}_*/"
echo "  Stage logs:    ${LOG_DIR}/*.log"
echo "================================================================"
