#!/usr/bin/env bash
# Round-19 paired evaluation driver.
#
# Iterates the (12 runs × 2 ckpts × 3 cfg_scales) eval matrix using
# eval_stage1_coarse_prior.py. **Single-GPU serial** mode — runs Plan A
# and S1-O sequentially on GPU0 per (seed, ckpt, cfg) iteration. Set
# GPU_ID below if a different GPU index is desired.
#
# Matrix:
#   12 runs  = 6 seeds × {Plan A, S1-O}
#   2 ckpts  = best-val (from training_summary.json) + ckpt-030000
#              (literature audit P1: best-val + secondary; ckpt-040000
#              dropped since 30k is already in the val-loss bottom region
#              per round19 val curves and adds ~50% wallclock for marginal
#              evidence)
#   3 cfgs   = cfg_scale_text ∈ {1.0, 2.5, 5.0}
#              (literature audit P0: standard motion-diffusion sweep)
#
# Per-config eval = 32 clips × 3 sample seeds = 96 samples × 1000-step
# DDPM ≈ 11-15 min on A6000. Matrix = 1 seed × 2 modes × 2 ckpts ×
# 3 cfgs = 12 configs. Wallclock single-GPU serial: 12 × 13 min ≈ ~2.6h.
#
# Prereqs:
#   - All 12 ckpts in runs/training/stage1_*_round19_seed{42..47}/
#   - training_summary.json per run (has best_val_ckpt_path)
#   - Selection JSON at the path SELECTION_JSON points to
#   - Conda env `piano` activated; PIANO_V18_CFG export not required
#     (eval reads cfg from the ckpt payload)
#
# Usage (from anywhere; script cd's to repo root):
#   bash scripts/stage_b_generator/run_round19_eval.sh
#
# Output:
#   analyses/round19_eval/<run_name>__<ckpt_label>__cfg<N>.json   (72 files)
#   analyses/round19_eval/eval_master.log                          (driver log)
#
# After completion: aggregate via
#   python scripts/stage_b_generator/aggregate_round19_paired_delta.py \
#       --eval-dir analyses/round19_eval/

set -euo pipefail
cd "$(dirname "$0")/../.."

# ----------- config -----------
GPU_ID=${GPU_ID:-0}                       # set via env: GPU_ID=1 bash run_round19_eval.sh
# Single training seed: Round-19 val-loss curves across all 6 seeds were
# nearly identical (paired Δ on val loss sign-consistent 6/6, std ≈ 0.046
# on Δ mean −0.32). Marginal value of testing all 6 training seeds with
# sampled metrics is low; eval seed 42 covers the dominant signal and
# saves 6× wallclock. The aggregator's sign-consistency gate is skipped
# when N_train_seeds < 2.
SEEDS=(42)
CFG_SCALES=(1.0 2.5 5.0)
# Ckpt labels: best_val (resolved from training_summary.json) + a fixed
# step-30000 fallback for robustness. We do NOT eval final.pt — Round-19
# val curves showed catastrophic over-training between step 35k and 100k.
CKPT_LABELS=(best_val ckpt-030000)

SELECTION_JSON="analyses/2026-05-20_round19_eval_selection.json"
EVAL_DIR="analyses/round19_eval"
MASTER_LOG="${EVAL_DIR}/eval_master.log"

CACHE_PLAN_A="cache/stage1_coarse_v1_full"
CACHE_S1O="cache/stage1_coarse_v1_objtraj_root0_world_round18_fix"

mkdir -p "${EVAL_DIR}"
echo "[round19-eval] repo root: $(pwd)" | tee "${MASTER_LOG}"
echo "[round19-eval] selection: ${SELECTION_JSON}" | tee -a "${MASTER_LOG}"
echo "[round19-eval] eval dir:  ${EVAL_DIR}" | tee -a "${MASTER_LOG}"
echo "[round19-eval] seeds:     ${SEEDS[*]}" | tee -a "${MASTER_LOG}"
echo "[round19-eval] ckpts:     ${CKPT_LABELS[*]}" | tee -a "${MASTER_LOG}"
echo "[round19-eval] cfgs:      ${CFG_SCALES[*]}" | tee -a "${MASTER_LOG}"
echo "[round19-eval] gpu:       cuda:${GPU_ID} (serial mode)" | tee -a "${MASTER_LOG}"
echo "" | tee -a "${MASTER_LOG}"

# Sanity: selection JSON must exist (run build_round19_eval_selection.py first).
if [[ ! -f "${SELECTION_JSON}" ]]; then
    echo "[round19-eval] ERROR — selection JSON missing: ${SELECTION_JSON}" \
        | tee -a "${MASTER_LOG}"
    echo "[round19-eval] generate it via:" | tee -a "${MASTER_LOG}"
    echo "  python scripts/stage_b_generator/build_round19_eval_selection.py \\" \
        | tee -a "${MASTER_LOG}"
    echo "    --cache-root ${CACHE_PLAN_A} --num-per-subset 8 --seed 42 \\" \
        | tee -a "${MASTER_LOG}"
    echo "    --output ${SELECTION_JSON}" | tee -a "${MASTER_LOG}"
    exit 1
fi

# ----------- per-config eval function -----------
# Args: gpu_id, run_dir, mode (s1a_cmc | s1o), seed, cfg, ckpt_label, cache_root
run_one_eval() {
    local GPU_ID="$1"
    local RUN_DIR="$2"
    local MODE="$3"
    local SEED="$4"
    local CFG="$5"
    local CKPT_LABEL="$6"
    local CACHE_ROOT="$7"

    # Resolve ckpt path. best_val comes from training_summary.json.
    local CKPT_PATH
    if [[ "${CKPT_LABEL}" == "best_val" ]]; then
        CKPT_PATH=$(python -c "
import json, sys
with open('${RUN_DIR}/training_summary.json') as f:
    s = json.load(f)
p = s.get('best_val_ckpt_path')
if p is None:
    sys.exit('no best_val_ckpt_path in training_summary.json')
print(p)
")
    else
        # Format like ckpt-030000.pt
        CKPT_PATH="${RUN_DIR}/${CKPT_LABEL}.pt"
    fi

    if [[ ! -f "${CKPT_PATH}" ]]; then
        echo "[round19-eval]   SKIP missing ckpt: ${CKPT_PATH}" \
            | tee -a "${MASTER_LOG}"
        return 0
    fi

    local RUN_NAME=$(basename "${RUN_DIR}")
    local CFG_TAG=$(printf "cfg%.1f" "${CFG}" | tr '.' '_')
    local OUT_TAG="${RUN_NAME}__${CKPT_LABEL}__${CFG_TAG}"
    local OUT_JSON="${EVAL_DIR}/${OUT_TAG}.json"

    if [[ -f "${OUT_JSON}" ]]; then
        echo "[round19-eval]   SKIP existing: ${OUT_JSON}" \
            | tee -a "${MASTER_LOG}"
        return 0
    fi

    echo "[round19-eval]   GPU${GPU_ID} ${OUT_TAG}  ckpt=${CKPT_PATH}" \
        | tee -a "${MASTER_LOG}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python scripts/stage_b_generator/eval_stage1_coarse_prior.py \
        --ckpt "${CKPT_PATH}" \
        --cache-root "${CACHE_ROOT}" \
        --selection-json "${SELECTION_JSON}" \
        --strict-selection \
        --tag "${OUT_TAG}" \
        --seeds 42,43,44 \
        --cfg-scale-text "${CFG}" \
        --inpaint-frame0 \
        --output-dir "${EVAL_DIR}" \
        >> "${EVAL_DIR}/${OUT_TAG}.stdout.log" 2>&1

    # Rename to predictable filename (eval_stage1 appends its own date prefix).
    local AUTO_NAME=$(ls -t "${EVAL_DIR}"/2026-*_stage1_eval_${OUT_TAG}.json 2>/dev/null | head -1 || true)
    if [[ -n "${AUTO_NAME}" && "${AUTO_NAME}" != "${OUT_JSON}" ]]; then
        mv "${AUTO_NAME}" "${OUT_JSON}"
    fi
}

# ----------- main matrix loop (single-GPU serial) -----------
for SEED in "${SEEDS[@]}"; do
    for CKPT_LABEL in "${CKPT_LABELS[@]}"; do
        for CFG in "${CFG_SCALES[@]}"; do
            echo "" | tee -a "${MASTER_LOG}"
            echo "===== seed=${SEED}  ckpt=${CKPT_LABEL}  cfg=${CFG} =====" \
                | tee -a "${MASTER_LOG}"

            RUN_PLAN_A="runs/training/stage1_s1a_cmc_round19_seed${SEED}"
            RUN_S1O="runs/training/stage1_s1o_round19_seed${SEED}"

            # Serial: Plan A then S1-O on the same GPU.
            set +e
            run_one_eval "${GPU_ID}" "${RUN_PLAN_A}" "s1a_cmc" "${SEED}" "${CFG}" \
                "${CKPT_LABEL}" "${CACHE_PLAN_A}"
            RC_A=$?
            run_one_eval "${GPU_ID}" "${RUN_S1O}" "s1o" "${SEED}" "${CFG}" \
                "${CKPT_LABEL}" "${CACHE_S1O}"
            RC_O=$?
            set -e
            if [[ ${RC_A} -ne 0 || ${RC_O} -ne 0 ]]; then
                echo "[round19-eval]   PAIR FAIL: Plan A rc=${RC_A}  S1-O rc=${RC_O}" \
                    | tee -a "${MASTER_LOG}"
                # Continue — partial results still useful; aggregator handles missing
            fi
        done
    done
done

echo "" | tee -a "${MASTER_LOG}"
echo "[round19-eval] ALL CONFIGS DISPATCHED — counting outputs:" \
    | tee -a "${MASTER_LOG}"
N_DONE=$(ls "${EVAL_DIR}"/stage1_*round19_seed*__*.json 2>/dev/null | wc -l)
N_EXPECT=$(( ${#SEEDS[@]} * 2 * ${#CKPT_LABELS[@]} * ${#CFG_SCALES[@]} ))
echo "[round19-eval] outputs = ${N_DONE} / expected ${N_EXPECT}" \
    | tee -a "${MASTER_LOG}"
echo "" | tee -a "${MASTER_LOG}"
echo "[round19-eval] aggregate via:" | tee -a "${MASTER_LOG}"
echo "  python scripts/stage_b_generator/aggregate_round19_paired_delta.py \\" \
    | tee -a "${MASTER_LOG}"
echo "    --eval-dir ${EVAL_DIR}" | tee -a "${MASTER_LOG}"
