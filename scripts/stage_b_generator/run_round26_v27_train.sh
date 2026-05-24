#!/usr/bin/env bash
# Round-26 v27 motion-faithful Stage-2 fine-tune launcher (DUAL-GPU).
#
# This is a single fine-tune run on the v25/R23 no-plan ckpt with the
# Round-26 motion-faithful loss patch
# (see analyses/2026-05-24_round26_v27_motion_faithful_patch.md).
#
# Wall-clock budget:
#   PREP    ~1 min   (translate config paths)
#   TRAIN   ~3–4 h   (80 epochs, fine-tune from R23 ckpt, SINGLE-GPU cuda:0,
#                     effective bs = 16 × 1 × 4 = 64; matches R23/v25 setting)
#   EVAL    ~50 min  (D2 + D3 on best_val.pt cuda:0 || final.pt cuda:1; dual-GPU)
#   TOTAL   ~4–5 h
#
# Note: training stage uses single GPU because PIANO's step_fn invokes
# `model.training_step(...)` (train_anchordiff.py:677) which under DDP
# bypasses gradient sync — DDP wrapping was never validated end-to-end in
# this repo. Eval scripts run as plain single-process Python and freely use
# either GPU via CUDA_VISIBLE_DEVICES, so eval-time dual-GPU is safe.
#
# Prerequisites on the Linux server:
#   1. git pull (must include the Round-26 commits)
#   2. v26/R23 ckpt present at:
#         runs/training/stageB_anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA/final.pt
#   3. Stage-1 S1-O ckpt present at:
#         runs/training/stage1_s1o_round20_seed42/final.pt
#   4. Stage-1 coarse cache (oracle Coarse-v1 + sampled S1-O):
#         cache/stage1_coarse_v1_full                                  (oracle norm)
#         cache/stage1_coarse_v1_objtraj_root0_world_round18_fix       (sampled-coarse)
#   5. Multimodal eval subset:
#         analyses/round25_multimodal_eval_subset.json
#      (created in Round-25 P0; should still be in repo / on server)
#
# Usage:
#   bash scripts/stage_b_generator/run_round26_v27_train.sh
#
# Skip-stage support (resume after a partial failure):
#   ROUND26_RESUME_FROM=eval bash <this script>     # skip PREP + TRAIN
#   ROUND26_RESUME_FROM=eval_final bash <this script> # only do final.pt eval
#
# Single-GPU fallback (cuda:0 only, no DDP, sequential eval):
#   ROUND26_SINGLE_GPU=1 bash <this script>

set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="runs/round26_v27_train"
mkdir -p "${LOG_DIR}"

V27_CFG_LOCAL="configs/training/anchordiff_v27_stage2_anchoraware_FULL_DATA_local.yaml"
V27_RUN_DIR="runs/training/stageB_anchordiff_v27_stage2_anchoraware_FULL_DATA"
R23_CKPT="runs/training/stageB_anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA/final.pt"
S1_CKPT="runs/training/stage1_s1o_round20_seed42/final.pt"
S1_CACHE="cache/stage1_coarse_v1_objtraj_root0_world_round18_fix"
EVAL_SUBSET="analyses/round25_multimodal_eval_subset.json"

SINGLE_GPU="${ROUND26_SINGLE_GPU:-0}"
RESUME_FROM="${ROUND26_RESUME_FROM:-}"

_should_skip() {
    # _should_skip <current_stage_index>
    local stages=(prep train eval eval_final)
    [[ -z "${RESUME_FROM}" ]] && return 1
    local target_idx=-1 current_idx=-1 i
    for ((i=0; i<${#stages[@]}; i++)); do
        [[ "${stages[i]}" == "$1" ]] && current_idx=$i
        [[ "${stages[i]}" == "${RESUME_FROM}" ]] && target_idx=$i
    done
    [[ $target_idx -lt 0 ]] && { echo "WARN: unknown ROUND26_RESUME_FROM=${RESUME_FROM}"; return 1; }
    [[ $current_idx -lt $target_idx ]]
}

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

# Background step (caller backgrounds with &). Writes ONLY to log.
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
# PREP: translate v27 config to server paths
# ============================================================
if _should_skip prep; then
    echo "[SKIP] PREP (ROUND26_RESUME_FROM=${RESUME_FROM})"
else
    run_step "prep_make_local_configs" \
        bash scripts/stage_b_generator/run_round26_make_local_configs.sh
fi

# ---------- preflight ----------
for F in "${V27_CFG_LOCAL}" "${R23_CKPT}" "${S1_CKPT}" "${EVAL_SUBSET}"; do
    if [[ ! -e "${F}" ]]; then
        echo "ERROR: missing prerequisite: ${F}"
        exit 1
    fi
done
if [[ ! -d "${S1_CACHE}" ]]; then
    echo "ERROR: missing Stage-1 cache: ${S1_CACHE}"
    exit 1
fi

# ============================================================
# TRAIN: v27 fine-tune from R23 ckpt
# ============================================================
if _should_skip train; then
    echo "[SKIP] TRAIN (ROUND26_RESUME_FROM=${RESUME_FROM})"
else
    # Training is single-GPU regardless of SINGLE_GPU env (see header note).
    # SINGLE_GPU only affects the EVAL stage (parallel vs sequential).
    echo "================================================================"
    echo "[$(date '+%F %T')] TRAIN v27 (SINGLE-GPU cuda:0, --num_processes 1)"
    echo "  effective batch size = bs(16) × num_proc(1) × accum(4) = 64"
    echo "================================================================"
    run_step "v27_train" \
        CUDA_VISIBLE_DEVICES=0 \
        conda run --no-capture-output -n piano accelerate launch \
            --num_processes 1 --mixed_precision bf16 \
            --main_process_port 29500 \
            src/piano/training/train_anchordiff.py \
            --config "${V27_CFG_LOCAL}"
fi

# Locate the trained ckpts.
V27_BEST="${V27_RUN_DIR}/best_val.pt"
V27_FINAL="${V27_RUN_DIR}/final.pt"
for F in "${V27_BEST}" "${V27_FINAL}"; do
    if [[ ! -f "${F}" ]]; then
        echo "WARN: expected ckpt missing after training: ${F}"
    fi
done

# ============================================================
# EVAL: D2 + D3 on best_val.pt (cuda:0) || best_val.pt or final.pt (cuda:1)
# ============================================================
# Layout (dual-GPU, both ckpts evaluated in parallel):
#   cuda:0  →  D2(best_val)  → D3(best_val)
#   cuda:1  →  D2(final)     → D3(final)
# If only one ckpt was produced (e.g. best_val == final), the second
# pair still runs and gives a redundant number — cheap safety.
if _should_skip eval; then
    echo "[SKIP] EVAL best_val (ROUND26_RESUME_FROM=${RESUME_FROM})"
else
    if [[ "${SINGLE_GPU}" == "1" ]]; then
        echo "================================================================"
        echo "[$(date '+%F %T')] EVAL on cuda:0 (sequential: D2 best, D3 best, D2 final, D3 final)"
        echo "================================================================"
        for TAG in best_val final; do
            CKPT="${V27_RUN_DIR}/${TAG}.pt"
            [[ -f "${CKPT}" ]] || { echo "[skip] missing ${CKPT}"; continue; }
            run_step "d2_v27_${TAG}" \
                conda run --no-capture-output -n piano python -u \
                    scripts/stage_b_generator/round25_d2_diversity_diagnostic.py \
                    --config "${V27_CFG_LOCAL}" --ckpt "${CKPT}" \
                    --selection-json "${EVAL_SUBSET}" \
                    --bucket val --n-samples 8 --cfg-scale 1.0 \
                    --output "analyses/round26_v27_d2_diversity_${TAG}.json"
            run_step "d3_v27_${TAG}" \
                conda run --no-capture-output -n piano python -u \
                    scripts/stage_b_generator/round25_d3_oracle_vs_sampled.py \
                    --config "${V27_CFG_LOCAL}" --ckpt "${CKPT}" \
                    --stage1-ckpt "${S1_CKPT}" --stage1-cache-root "${S1_CACHE}" \
                    --selection-json "${EVAL_SUBSET}" \
                    --bucket val --cfg-scale 1.0 --cfg-scale-stage1 1.0 --seed 42 \
                    --output "analyses/round26_v27_d3_oracle_vs_sampled_${TAG}.json"
        done
    else
        echo "================================================================"
        echo "[$(date '+%F %T')] EVAL DUAL-GPU: cuda:0 best_val (D2 then D3) || cuda:1 final (D2 then D3)"
        echo "================================================================"

        # Wrap "D2 then D3" into a single bash invocation so the GPU stays
        # held by ONE process tree (avoids cross-GPU contention).
        # Skipping happens inside the inner script.
        _eval_on_gpu() {
            local TAG="$1"
            local GPU="$2"
            local CKPT="${V27_RUN_DIR}/${TAG}.pt"
            if [[ ! -f "${CKPT}" ]]; then
                echo "[$(date '+%F %T')] [skip eval_${TAG}] missing ${CKPT}" > "${LOG_DIR}/d2_v27_${TAG}.log"
                return 0
            fi
            {
                echo "================================================================"
                echo "[$(date '+%F %T')] BG-EVAL ${TAG} on cuda:${GPU}"
                echo "================================================================"
                CUDA_VISIBLE_DEVICES="${GPU}" \
                PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
                    conda run --no-capture-output -n piano python -u \
                        scripts/stage_b_generator/round25_d2_diversity_diagnostic.py \
                        --config "${V27_CFG_LOCAL}" --ckpt "${CKPT}" \
                        --selection-json "${EVAL_SUBSET}" \
                        --bucket val --n-samples 8 --cfg-scale 1.0 \
                        --output "analyses/round26_v27_d2_diversity_${TAG}.json" 2>&1
                local EX_D2=$?
                echo "[$(date '+%F %T')] BG-EVAL ${TAG} D2 exit=${EX_D2}"
                CUDA_VISIBLE_DEVICES="${GPU}" \
                PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
                    conda run --no-capture-output -n piano python -u \
                        scripts/stage_b_generator/round25_d3_oracle_vs_sampled.py \
                        --config "${V27_CFG_LOCAL}" --ckpt "${CKPT}" \
                        --stage1-ckpt "${S1_CKPT}" --stage1-cache-root "${S1_CACHE}" \
                        --selection-json "${EVAL_SUBSET}" \
                        --bucket val --cfg-scale 1.0 --cfg-scale-stage1 1.0 --seed 42 \
                        --output "analyses/round26_v27_d3_oracle_vs_sampled_${TAG}.json" 2>&1
                local EX_D3=$?
                echo "[$(date '+%F %T')] BG-EVAL ${TAG} D3 exit=${EX_D3}"
            } > "${LOG_DIR}/eval_${TAG}.log" 2>&1
        }

        _eval_on_gpu best_val 0 &
        PID_BEST=$!
        _eval_on_gpu final 1 &
        PID_FINAL=$!
        echo "    PID best_val=${PID_BEST}  final=${PID_FINAL}"
        echo "    follow: tail -f ${LOG_DIR}/eval_best_val.log ${LOG_DIR}/eval_final.log"

        EX_BEST=0; EX_FINAL=0
        wait $PID_BEST || EX_BEST=$?
        wait $PID_FINAL || EX_FINAL=$?
        if [[ $EX_BEST -ne 0 || $EX_FINAL -ne 0 ]]; then
            echo "WARN: EVAL failures (best=${EX_BEST} final=${EX_FINAL}); continuing to PACK anyway."
        fi
        echo "[$(date '+%F %T')] EVAL DONE (best=${EX_BEST} final=${EX_FINAL})"
    fi
fi

# ============================================================
# PACK: tar all output files into a single tarball
# ============================================================
echo
echo "================================================================"
echo "[$(date '+%F %T')] PACK results"
echo "================================================================"
run_step "pack" \
    bash scripts/stage_b_generator/round26_pack_results.sh

echo
echo "================================================================"
echo "Round-26 v27 train+eval complete."
echo "Outputs:"
echo "  v27 ckpts:   ${V27_RUN_DIR}/{best_val,final,epoch_*}.pt  (server only, NOT in tarball)"
echo "  v27 metrics: ${V27_RUN_DIR}/metrics.jsonl"
echo "  D2 eval:     analyses/round26_v27_d2_diversity_{best_val,final}.{json,md}"
echo "  D3 eval:     analyses/round26_v27_d3_oracle_vs_sampled_{best_val,final}.{json,md}"
echo "  Stage logs:  ${LOG_DIR}/*.log"
echo "  Tarball:     round26_results_*.tar.gz at project root (see PACK output above)"
echo "================================================================"
