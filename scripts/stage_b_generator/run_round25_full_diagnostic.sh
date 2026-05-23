#!/usr/bin/env bash
# Round-25 P0 full diagnostic bundle launcher (server side).
#
# Runs the entire 5-experiment diagnostic in sequence on a single GPU,
# logging each stage to runs/round25_diagnostic/. Designed for the
# 2× A6000 server; uses cuda:0 by default. To run on cuda:1:
#   CUDA_VISIBLE_DEVICES=1 bash <this script>
#
# Prerequisites:
#   1. git pull to latest commit
#   2. bash scripts/stage_b_generator/run_round25_make_local_configs.sh
#      (sed Windows → Linux dataset paths)
#   3. v26 ckpt exists:
#      runs/training/stageB_anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA/final.pt
#   4. Stage-1 ckpt exists:
#      runs/training/stage1_s1o_round20_seed42/final.pt
#   5. Stage-1 cache exists:
#      cache/stage1_coarse_v1_objtraj_root0_world_round18_fix/
#   6. Stage-2 v26 cache exists:
#      cache/stage1_coarse_v1_full/
#
# Stages (~all on cuda:0, sequential):
#   D1   propose val + train multimodal candidates (~5 min)
#   D1c  curate candidates → final subset JSONs (~30s)
#   D4i  build D4 subset_indices_file JSONs (~30s)
#   D2   v26 multi-sample diversity, 48 clips × N=8 samples (~15-30 min A6000)
#   D3   v26 oracle-vs-sampled Stage-1, 48 clips × 2 samples each (~10-20 min)
#   D4-8 v26 overfit 8 clips × 600 epochs (~30 min)
#   D4-16 v26 overfit 16 clips × 600 epochs (~60 min)
#   D5-V0 v26 + endpoint_weight=1.0 × 5 epochs ~74 clips (~30 min)
#   D5-V1 v26 + endpoint_weight=2.0 × 5 epochs ~74 clips (~30 min)
#   D5-V2 v26 + endpoint_weight=5.0 × 5 epochs ~74 clips (~30 min)
#
# Total estimated wall: ~4-6 hours on a single A6000.

set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="runs/round25_diagnostic"
mkdir -p "${LOG_DIR}"

V26_CFG_LOCAL="configs/training/anchordiff_v26_FULL_DATA_local.yaml"
V26_CKPT="runs/training/stageB_anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA/final.pt"
S1_CKPT="runs/training/stage1_s1o_round20_seed42/final.pt"
S1_CACHE="cache/stage1_coarse_v1_objtraj_root0_world_round18_fix"

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

run_step() {
    local NAME="$1"; shift
    local LOG="${LOG_DIR}/${NAME}.log"
    local T0
    T0=$(date +%s)
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] START ${NAME}"
    echo "    cmd: $*"
    echo "    log: ${LOG}"
    echo "================================================================"
    # Use unbuffered Python so progress lines stream to log immediately
    # (per the local-pipeline-trace feedback: avoid buffered stdout on
    # long-running diagnostics).
    PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$@" 2>&1 | tee "${LOG}"
    local T1
    T1=$(date +%s)
    echo "[$(date '+%F %T')] DONE ${NAME} in $((T1-T0))s"
}

# ============================================================
# D1: propose val + train multimodal candidates
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
# D1c: curate val + train final subsets
# ============================================================
run_step "d1_curate" \
    conda run --no-capture-output -n piano python -u \
        scripts/stage_b_generator/round25_curate_subsets.py

# ============================================================
# D4 indices: convert train selection → trainer subset_indices_file
# ============================================================
run_step "d4_build_indices_8" \
    conda run --no-capture-output -n piano python -u \
        scripts/stage_b_generator/round25_d4_build_subset_indices.py \
        --config "${V26_CFG_LOCAL}" \
        --train-selection-json analyses/round25_d4_train_selection.json \
        --n 8 --output analyses/round25_d4_indices_8.json

run_step "d4_build_indices_16" \
    conda run --no-capture-output -n piano python -u \
        scripts/stage_b_generator/round25_d4_build_subset_indices.py \
        --config "${V26_CFG_LOCAL}" \
        --train-selection-json analyses/round25_d4_train_selection.json \
        --n 16 --output analyses/round25_d4_indices_16.json

# ============================================================
# D2: multi-sample diversity (read-only on v26)
# ============================================================
run_step "d2_diversity" \
    conda run --no-capture-output -n piano python -u \
        scripts/stage_b_generator/round25_d2_diversity_diagnostic.py \
        --config "${V26_CFG_LOCAL}" \
        --ckpt "${V26_CKPT}" \
        --selection-json analyses/round25_multimodal_eval_subset.json \
        --bucket val --n-samples 8 --cfg-scale 1.0 \
        --output analyses/round25_d2_diversity_stats.json

# ============================================================
# D3: oracle vs sampled Stage-1 (read-only on v26 + Stage-1)
# ============================================================
run_step "d3_oracle_vs_sampled" \
    conda run --no-capture-output -n piano python -u \
        scripts/stage_b_generator/round25_d3_oracle_vs_sampled.py \
        --config "${V26_CFG_LOCAL}" \
        --ckpt "${V26_CKPT}" \
        --stage1-ckpt "${S1_CKPT}" \
        --stage1-cache-root "${S1_CACHE}" \
        --selection-json analyses/round25_multimodal_eval_subset.json \
        --bucket val --cfg-scale 1.0 --cfg-scale-stage1 1.0 --seed 42 \
        --output analyses/round25_d3_oracle_vs_sampled.json

# ============================================================
# D4: tiny overfit (8 and 16 clips)
# ============================================================
# These use the existing train_anchordiff entry point.
run_step "d4_overfit8" \
    conda run --no-capture-output -n piano accelerate launch \
        --num_processes 1 \
        --mixed_precision bf16 \
        src/piano/training/train_anchordiff.py \
        --config configs/training/anchordiff_v26_d4_overfit8_local.yaml

run_step "d4_overfit16" \
    conda run --no-capture-output -n piano accelerate launch \
        --num_processes 1 \
        --mixed_precision bf16 \
        src/piano/training/train_anchordiff.py \
        --config configs/training/anchordiff_v26_d4_overfit16_local.yaml

# ============================================================
# D5: loss-weight smoke (V0 / V1 / V2)
# ============================================================
for V in v0_baseline v1_hand2x_foot2x v2_hand5x_foot5x; do
    run_step "d5_${V}" \
        conda run --no-capture-output -n piano accelerate launch \
            --num_processes 1 \
            --mixed_precision bf16 \
            src/piano/training/train_anchordiff.py \
            --config "configs/training/anchordiff_v26_d5_${V}_local.yaml"
done

echo
echo "================================================================"
echo "Round-25 full diagnostic bundle complete."
echo "Outputs:"
echo "  D1 candidates: analyses/round25_multimodal_candidates_{val,train}.json"
echo "  D1 subsets:    analyses/round25_multimodal_eval_subset.json"
echo "                 analyses/round25_d4_train_selection.json"
echo "  D2 stats:      analyses/round25_d2_diversity_stats.{json,md}"
echo "  D3 stats:      analyses/round25_d3_oracle_vs_sampled.{json,md}"
echo "  D4 runs:       runs/training/stageB_anchordiff_v26_d4_overfit{8,16}/"
echo "  D5 runs:       runs/training/stageB_anchordiff_v26_d5_{v0,v1,v2}_*/"
echo "  Stage logs:    ${LOG_DIR}/*.log"
echo "================================================================"
