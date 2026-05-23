#!/usr/bin/env bash
# Round-25 P1 launcher — runs the post-P0 investigations:
#
#   P1.2  D5 redo at 30 epoch (V0 baseline + V1 hand×2 + V2 hand×5)
#         Tests H2 (loss imbalance) properly.
#   P1.1  Stage-1 sampler audit (cfg_scale × num_steps matrix)
#         Tests whether the D3 +12.7cm gap is fixable by inference
#         config alone, or requires Stage-1 retrain.
#
# H_new dynamics-bias audit (P1.3) is NOT in this launcher because
# each candidate (min-SNR γ, root-vel weight, FK-pos weighting) is a
# separate D4-style overfit experiment best run manually after
# inspecting each result.
#
# Usage (server, 2× A6000):
#   git pull
#   bash scripts/stage_b_generator/run_round25_make_local_configs.sh
#   bash scripts/stage_b_generator/run_round25_p1.sh
#
# Total wall: ~1-2 hours (vs ~13 days of the rejected original P1).

set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="runs/round25_diagnostic"
mkdir -p "${LOG_DIR}"

V26_CFG_LOCAL="configs/training/anchordiff_v26_FULL_DATA_local.yaml"
S1_CKPT="runs/training/stage1_s1o_round20_seed42/final.pt"
S1_CACHE="cache/stage1_coarse_v1_objtraj_root0_world_round18_fix"

# Sanity check the prerequisites.
for F in "${V26_CFG_LOCAL}" "${S1_CKPT}"; do
    if [[ ! -e "${F}" ]]; then
        echo "ERROR: missing prerequisite: ${F}"
        exit 1
    fi
done
if [[ ! -d "${S1_CACHE}" ]]; then
    echo "ERROR: missing Stage-1 cache: ${S1_CACHE}"
    exit 1
fi
if [[ ! -e "analyses/round25_multimodal_eval_subset.json" ]]; then
    echo "ERROR: D1 eval subset missing. Run run_round25_full_diagnostic.sh first."
    exit 1
fi

run_step_bg() {
    local NAME="$1"; shift
    local GPU="$1"; shift
    local LOG="${LOG_DIR}/${NAME}.log"
    local T0
    T0=$(date +%s)
    {
        echo "================================================================"
        echo "[$(date '+%F %T')] BG-START ${NAME} on cuda:${GPU}"
        echo "    log: ${LOG}"
        echo "================================================================"
        CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$@" 2>&1
        local T1
        T1=$(date +%s)
        echo "[$(date '+%F %T')] BG-DONE ${NAME} in $((T1-T0))s"
    } > "${LOG}" 2>&1
}

# ============================================================
# PHASE-E: D5 redo at 30 epoch — V0 (cuda:0) || V1 (cuda:1)
# ============================================================
echo
echo "================================================================"
echo "[$(date '+%F %T')] PHASE-E: D5-30ep V0 (cuda:0) || V1 (cuda:1)"
echo "================================================================"

run_step_bg "d5_v0_baseline_30ep" 0 \
    conda run --no-capture-output -n piano accelerate launch \
        --num_processes 1 --mixed_precision bf16 \
        --main_process_port 29500 \
        src/piano/training/train_anchordiff.py \
        --config configs/training/anchordiff_v26_d5_v0_baseline_30ep_local.yaml &
PID_V0=$!

run_step_bg "d5_v1_hand2x_foot2x_30ep" 1 \
    conda run --no-capture-output -n piano accelerate launch \
        --num_processes 1 --mixed_precision bf16 \
        --main_process_port 29501 \
        src/piano/training/train_anchordiff.py \
        --config configs/training/anchordiff_v26_d5_v1_hand2x_foot2x_30ep_local.yaml &
PID_V1=$!

echo "    PID V0=${PID_V0}  V1=${PID_V1}"
EX_V0=0; EX_V1=0
wait $PID_V0 || EX_V0=$?
wait $PID_V1 || EX_V1=$?
echo "[$(date '+%F %T')] PHASE-E DONE (V0=${EX_V0} V1=${EX_V1})"

# ============================================================
# PHASE-F: D5 V2 30ep (cuda:0) || Stage-1 sampler audit (cuda:1)
# ============================================================
echo
echo "================================================================"
echo "[$(date '+%F %T')] PHASE-F: D5-V2 30ep (cuda:0) || P1.1 sampler audit (cuda:1)"
echo "================================================================"

run_step_bg "d5_v2_hand5x_foot5x_30ep" 0 \
    conda run --no-capture-output -n piano accelerate launch \
        --num_processes 1 --mixed_precision bf16 \
        --main_process_port 29500 \
        src/piano/training/train_anchordiff.py \
        --config configs/training/anchordiff_v26_d5_v2_hand5x_foot5x_30ep_local.yaml &
PID_V2=$!

run_step_bg "p11_stage1_sampler_audit" 1 \
    conda run --no-capture-output -n piano python -u \
        scripts/stage_b_generator/round25_p11_stage1_sampler_audit.py \
        --config "${V26_CFG_LOCAL}" \
        --stage1-ckpt "${S1_CKPT}" \
        --stage1-cache-root "${S1_CACHE}" \
        --selection-json analyses/round25_multimodal_eval_subset.json \
        --bucket val \
        --cfg-scale-grid 0.5,1.0,2.0,4.0 \
        --num-steps-grid 50,100,200,1000 \
        --seeds 42,43,44 \
        --output analyses/round25_p11_stage1_sampler_audit.json &
PID_P11=$!

echo "    PID V2=${PID_V2}  P11-audit=${PID_P11}"
EX_V2=0; EX_P11=0
wait $PID_V2 || EX_V2=$?
wait $PID_P11 || EX_P11=$?
echo "[$(date '+%F %T')] PHASE-F DONE (V2=${EX_V2} P11=${EX_P11})"

echo
echo "================================================================"
echo "Round-25 P1.1 + P1.2 complete."
echo "Outputs:"
echo "  D5 redo metrics:  runs/training/stageB_anchordiff_v26_d5_v{0,1,2}_*_30ep/metrics.jsonl"
echo "  P1.1 audit:       analyses/round25_p11_stage1_sampler_audit.{json,md}"
echo "  Stage logs:       ${LOG_DIR}/d5_*_30ep.log + ${LOG_DIR}/p11_stage1_sampler_audit.log"
echo
echo "Pack + transfer to local with:"
echo "  bash scripts/stage_b_generator/round25_pack_results.sh"
echo "================================================================"
