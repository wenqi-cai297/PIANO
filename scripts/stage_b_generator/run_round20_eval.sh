#!/usr/bin/env bash
# Round-20 paired evaluation driver.
#
# Iterates the (2 runs × 2 ckpts × 3 cfg_scales) eval matrix using
# eval_stage1_coarse_prior.py. **Single-GPU serial** mode — runs Plan A
# and S1-O sequentially on GPU0 per (seed, ckpt, cfg) iteration. Set
# GPU_ID below if a different GPU index is desired.
#
# Matrix:
#   2 runs   = 1 seed × {Plan A, S1-O}
#              (Round-20 is method-iteration mode — single seed 42,
#              multi-seed stability sweep deferred until method is locked)
#   2 ckpts  = best-val (= final.pt at step 40000 in Round-20 schedule)
#              + ckpt-030000 (mid-plateau control; val_live at 30k is
#              within noise of 40k per training_summary.json)
#   3 cfgs   = cfg_scale_text ∈ {1.0, 2.5, 5.0}
#              (standard motion-diffusion CFG sweep)
#
# Per-config eval = 32 clips × 3 sample seeds = 96 samples × 1000-step
# DDPM ≈ 11-15 min on A6000. Matrix = 1 seed × 2 modes × 2 ckpts ×
# 3 cfgs = 12 configs. Wallclock single-GPU serial: 12 × 13 min ≈ ~2.6h.
#
# Selection JSON: reuses the Round-19 selection (same Plan A cache,
# same manifest_val.jsonl, mode-agnostic clip picks) so Round-19 vs
# Round-20 numbers are directly comparable on identical clips.
#
# Prereqs:
#   - 2 ckpts in runs/training/stage1_*_round20_seed42/
#       * final.pt (best_val per training_summary.json)
#       * ckpt-030000.pt
#   - training_summary.json per run (has best_val_ckpt_path)
#   - Selection JSON at the path SELECTION_JSON points to (Round-19 reuse)
#   - Conda env `piano` activated
#
# Usage (from anywhere; script cd's to repo root):
#   bash scripts/stage_b_generator/run_round20_eval.sh
#
# Output:
#   analyses/round20_eval/stage1_<mode>_round20_seed42__<ckpt>__cfg<X_Y>.json (12 files)
#   analyses/round20_eval/eval_master.log                                      (driver log)
#
# After completion: aggregate via
#   python scripts/stage_b_generator/aggregate_round19_paired_delta.py \
#       --eval-dir analyses/round20_eval/ \
#       --output-prefix analyses/2026-05-XX_round20_paired_delta

set -euo pipefail
cd "$(dirname "$0")/../.."

# ----------- config -----------
GPU_ID=${GPU_ID:-0}                       # env override: GPU_ID=1 bash run_round20_eval.sh
SEEDS=(42)
CFG_SCALES=(1.0 2.5 5.0)
# Ckpt labels: best_val (resolved from training_summary.json → final.pt)
# + ckpt-030000 (mid-plateau control). Round-20 has clean save cadence
# every 5k so both ckpts are guaranteed exact-step matches.
CKPT_LABELS=(best_val ckpt-030000)

SELECTION_JSON="analyses/2026-05-20_round19_eval_selection.json"
EVAL_DIR="analyses/round20_eval"
MASTER_LOG="${EVAL_DIR}/eval_master.log"

CACHE_PLAN_A="cache/stage1_coarse_v1_full"
CACHE_S1O="cache/stage1_coarse_v1_objtraj_root0_world_round18_fix"

mkdir -p "${EVAL_DIR}"
echo "[round20-eval] repo root: $(pwd)" | tee "${MASTER_LOG}"
echo "[round20-eval] selection: ${SELECTION_JSON}" | tee -a "${MASTER_LOG}"
echo "[round20-eval] eval dir:  ${EVAL_DIR}" | tee -a "${MASTER_LOG}"
echo "[round20-eval] seeds:     ${SEEDS[*]}" | tee -a "${MASTER_LOG}"
echo "[round20-eval] ckpts:     ${CKPT_LABELS[*]}" | tee -a "${MASTER_LOG}"
echo "[round20-eval] cfgs:      ${CFG_SCALES[*]}" | tee -a "${MASTER_LOG}"
echo "[round20-eval] gpu:       cuda:${GPU_ID} (serial mode)" | tee -a "${MASTER_LOG}"
echo "" | tee -a "${MASTER_LOG}"

# Sanity: selection JSON must exist (Round-19 selection reused).
if [[ ! -f "${SELECTION_JSON}" ]]; then
    echo "[round20-eval] ERROR — selection JSON missing: ${SELECTION_JSON}" \
        | tee -a "${MASTER_LOG}"
    echo "[round20-eval] generate it via:" | tee -a "${MASTER_LOG}"
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

    # Resolve ckpt path. best_val comes from training_summary.json. The
    # Round-20 trainer writes best_val_ckpt_path = final.pt with
    # best_val_ckpt_exact=true (val=save cadence aligned at 5k), so the
    # nearest-fallback branch is normally unused — kept for robustness.
    local CKPT_PATH
    if [[ "${CKPT_LABEL}" == "best_val" ]]; then
        CKPT_PATH=$(python -c "
import json, sys
with open('${RUN_DIR}/training_summary.json') as f:
    s = json.load(f)
p = s.get('best_val_ckpt_path')
if p is None:
    p = s.get('best_val_nearest_ckpt_path')
    if p is None:
        sys.exit(
            'no best_val_ckpt_path AND no best_val_nearest_ckpt_path '
            'in training_summary.json'
        )
    print(p)
    print('[round20-eval][nearest-fallback] using nearest periodic ckpt: '
          + str(p), file=sys.stderr)
else:
    print(p)
")
    else
        # Format like ckpt-030000.pt
        CKPT_PATH="${RUN_DIR}/${CKPT_LABEL}.pt"
    fi

    if [[ ! -f "${CKPT_PATH}" ]]; then
        echo "[round20-eval]   SKIP missing ckpt: ${CKPT_PATH}" \
            | tee -a "${MASTER_LOG}"
        return 0
    fi

    local RUN_NAME=$(basename "${RUN_DIR}")
    local CFG_TAG=$(printf "cfg%.1f" "${CFG}" | tr '.' '_')
    local OUT_TAG="${RUN_NAME}__${CKPT_LABEL}__${CFG_TAG}"
    local OUT_JSON="${EVAL_DIR}/${OUT_TAG}.json"

    if [[ -f "${OUT_JSON}" ]]; then
        echo "[round20-eval]   SKIP existing: ${OUT_JSON}" \
            | tee -a "${MASTER_LOG}"
        return 0
    fi

    echo "[round20-eval]   GPU${GPU_ID} ${OUT_TAG}  ckpt=${CKPT_PATH}" \
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

            RUN_PLAN_A="runs/training/stage1_s1a_cmc_round20_seed${SEED}"
            RUN_S1O="runs/training/stage1_s1o_round20_seed${SEED}"

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
                echo "[round20-eval]   PAIR FAIL: Plan A rc=${RC_A}  S1-O rc=${RC_O}" \
                    | tee -a "${MASTER_LOG}"
                # Continue — partial results still useful; aggregator handles missing
            fi
        done
    done
done

echo "" | tee -a "${MASTER_LOG}"
echo "[round20-eval] ALL CONFIGS DISPATCHED — counting outputs:" \
    | tee -a "${MASTER_LOG}"
N_DONE=$(ls "${EVAL_DIR}"/stage1_*round20_seed*__*.json 2>/dev/null | wc -l)
N_EXPECT=$(( ${#SEEDS[@]} * 2 * ${#CKPT_LABELS[@]} * ${#CFG_SCALES[@]} ))
echo "[round20-eval] outputs = ${N_DONE} / expected ${N_EXPECT}" \
    | tee -a "${MASTER_LOG}"
echo "" | tee -a "${MASTER_LOG}"
echo "[round20-eval] aggregate via:" | tee -a "${MASTER_LOG}"
echo "  python scripts/stage_b_generator/aggregate_round19_paired_delta.py \\" \
    | tee -a "${MASTER_LOG}"
echo "    --eval-dir ${EVAL_DIR} \\" | tee -a "${MASTER_LOG}"
echo "    --output-prefix analyses/2026-05-XX_round20_paired_delta" \
    | tee -a "${MASTER_LOG}"
