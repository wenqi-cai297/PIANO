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
# Resume support: set ROUND25_RESUME_FROM to skip stages already done.
#   ROUND25_RESUME_FROM=prep_b  → skip PREP-A (d1 propose)
#   ROUND25_RESUME_FROM=phase_a → skip PREP-A + PREP-B
#   ROUND25_RESUME_FROM=phase_b → skip PREP-A + PREP-B + PHASE-A
#   ROUND25_RESUME_FROM=phase_c → skip everything before D5 (saves D4 retraining)
#   ROUND25_RESUME_FROM=phase_d → skip everything before D5-V2
RESUME_FROM="${ROUND25_RESUME_FROM:-}"

_should_skip() {
    # _should_skip <current_stage_index>
    # Returns 0 (skip) if RESUME_FROM is set and points later than current.
    local stages=(prep_a prep_b phase_a phase_b phase_c phase_d)
    [[ -z "${RESUME_FROM}" ]] && return 1
    local target_idx=-1 current_idx=-1 i
    for ((i=0; i<${#stages[@]}; i++)); do
        [[ "${stages[i]}" == "$1" ]] && current_idx=$i
        [[ "${stages[i]}" == "${RESUME_FROM}" ]] && target_idx=$i
    done
    [[ $target_idx -lt 0 ]] && { echo "WARN: unknown ROUND25_RESUME_FROM=${RESUME_FROM}"; return 1; }
    [[ $current_idx -lt $target_idx ]]
}

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
if _should_skip prep_a; then
    echo "[SKIP] PREP-A (ROUND25_RESUME_FROM=${RESUME_FROM})"
else
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
fi

# ============================================================
# PREP-B: D1 curate + D4 indices (sequential, fast)
# ============================================================
if _should_skip prep_b; then
    echo "[SKIP] PREP-B (ROUND25_RESUME_FROM=${RESUME_FROM})"
else
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
fi

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

    if _should_skip phase_a; then
        echo "[SKIP] PHASE-A (ROUND25_RESUME_FROM=${RESUME_FROM})"
    else
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
    EX_D2=0; EX_D3=0
    wait $PID_D2 || EX_D2=$?
    wait $PID_D3 || EX_D3=$?
    if [[ $EX_D2 -ne 0 || $EX_D3 -ne 0 ]]; then
        echo "WARN: PHASE-A failures (D2=${EX_D2} D3=${EX_D3}); continuing anyway."
    fi
    echo "[$(date '+%F %T')] PHASE-A DONE (D2=${EX_D2} D3=${EX_D3})"
    fi  # end skip-guard phase_a

    if _should_skip phase_b; then
        echo "[SKIP] PHASE-B (ROUND25_RESUME_FROM=${RESUME_FROM})"
    else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE-B: D4-8 (cuda:0) || D4-16 (cuda:1)"
    echo "================================================================"
    # PORT FIX: parallel accelerate launches default to TCP port 29500
    # for rendezvous → second one EADDRINUSE-crashes. Give each its own port.
    run_step_bg "d4_overfit8" 0 \
        conda run --no-capture-output -n piano accelerate launch \
            --num_processes 1 --mixed_precision bf16 \
            --main_process_port 29500 \
            src/piano/training/train_anchordiff.py \
            --config configs/training/anchordiff_v26_d4_overfit8_local.yaml &
    PID_D4_8=$!

    run_step_bg "d4_overfit16" 1 \
        conda run --no-capture-output -n piano accelerate launch \
            --num_processes 1 --mixed_precision bf16 \
            --main_process_port 29501 \
            src/piano/training/train_anchordiff.py \
            --config configs/training/anchordiff_v26_d4_overfit16_local.yaml &
    PID_D4_16=$!

    echo "    PID D4-8=${PID_D4_8}  D4-16=${PID_D4_16}"
    # Robust wait: capture individual exit codes, do NOT abort the
    # pipeline if a single stage failed (set -e + plain `wait` would).
    EX_D4_8=0; EX_D4_16=0
    wait $PID_D4_8 || EX_D4_8=$?
    wait $PID_D4_16 || EX_D4_16=$?
    if [[ $EX_D4_8 -ne 0 || $EX_D4_16 -ne 0 ]]; then
        echo "WARN: PHASE-B failures (D4-8=${EX_D4_8} D4-16=${EX_D4_16}); continuing to PHASE-C anyway."
    fi
    echo "[$(date '+%F %T')] PHASE-B DONE (D4-8=${EX_D4_8} D4-16=${EX_D4_16})"
    fi  # end skip-guard phase_b

    if _should_skip phase_c; then
        echo "[SKIP] PHASE-C (ROUND25_RESUME_FROM=${RESUME_FROM})"
    else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE-C: D5-V0 (cuda:0) || D5-V1 (cuda:1)"
    echo "================================================================"
    run_step_bg "d5_v0_baseline" 0 \
        conda run --no-capture-output -n piano accelerate launch \
            --num_processes 1 --mixed_precision bf16 \
            --main_process_port 29500 \
            src/piano/training/train_anchordiff.py \
            --config configs/training/anchordiff_v26_d5_v0_baseline_local.yaml &
    PID_V0=$!

    run_step_bg "d5_v1_hand2x_foot2x" 1 \
        conda run --no-capture-output -n piano accelerate launch \
            --num_processes 1 --mixed_precision bf16 \
            --main_process_port 29501 \
            src/piano/training/train_anchordiff.py \
            --config configs/training/anchordiff_v26_d5_v1_hand2x_foot2x_local.yaml &
    PID_V1=$!

    echo "    PID V0=${PID_V0}  V1=${PID_V1}"
    EX_V0=0; EX_V1=0
    wait $PID_V0 || EX_V0=$?
    wait $PID_V1 || EX_V1=$?
    if [[ $EX_V0 -ne 0 || $EX_V1 -ne 0 ]]; then
        echo "WARN: PHASE-C failures (V0=${EX_V0} V1=${EX_V1}); continuing to PHASE-D anyway."
    fi
    echo "[$(date '+%F %T')] PHASE-C DONE (V0=${EX_V0} V1=${EX_V1})"
    fi  # end skip-guard phase_c

    if _should_skip phase_d; then
        echo "[SKIP] PHASE-D (ROUND25_RESUME_FROM=${RESUME_FROM})"
    else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE-D: D5-V2 (cuda:0) solo"
    echo "================================================================"
    # V2 solo on cuda:0; default port 29500 is fine since nothing else
    # is running. Foreground tee here.
    EX_V2=0
    run_step "d5_v2_hand5x_foot5x" \
        conda run --no-capture-output -n piano accelerate launch \
            --num_processes 1 --mixed_precision bf16 \
            --main_process_port 29500 \
            src/piano/training/train_anchordiff.py \
            --config configs/training/anchordiff_v26_d5_v2_hand5x_foot5x_local.yaml \
        || EX_V2=$?
    if [[ $EX_V2 -ne 0 ]]; then
        echo "WARN: PHASE-D failed (V2=${EX_V2})."
    fi
    fi  # end skip-guard phase_d
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
