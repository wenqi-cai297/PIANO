#!/usr/bin/env bash
# Round-26 v27 per-part anchor diagnostic launcher.
#
# Runs scripts/stage_b_generator/anchor_realization_diagnostic.py on three
# ckpts using the Round-25 multimodal eval subset (48 clips, same selection
# as D2/D3 — directly comparable):
#
#   cuda:0  →  v27 final.pt     →  analyses/round26_v27_anchor_diag_final/
#   cuda:1  →  v27 best_val.pt  →  analyses/round26_v27_anchor_diag_best_val/
#   cuda:0  →  R23 no-plan ckpt →  analyses/round26_r23_anchor_diag_48clips/
#
# The R23 pass re-runs the Round-24 anchor diagnostic on the same 48-clip
# subset (Round-24 used the older 32-clip R19 selection) so v27 vs R23
# per-part deltas are computed on identical clips.
#
# Wall-clock on 2× A6000:
#   PAIR-1 (v27 final cuda:0 || v27 best_val cuda:1)   ~30 min
#   PAIR-2 (R23 baseline on cuda:0, cuda:1 idle)       ~30 min
#   PACK                                                ~10 s
#   TOTAL                                               ~60 min
#
# Each run produces:
#   <output_dir>/anchor_stats.json
#   <output_dir>/anchor_summary.md     (per-part: l_hand, r_hand, l_foot, r_foot, pelvis)
#   <output_dir>/distance_scatter.png
#   <output_dir>/per_part_histogram.png
#   <output_dir>/per_subset_bars.png
#
# Prerequisites (server):
#   git pull (must have v27 train ckpts written)
#   v27 ckpts at runs/training/stageB_anchordiff_v27_stage2_anchoraware_FULL_DATA/{best_val,final}.pt
#   R23 ckpt at runs/training/stageB_anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA/final.pt
#   analyses/round25_multimodal_eval_subset.json
#   configs/training/anchordiff_v27_stage2_anchoraware_FULL_DATA_local.yaml  (from prior run)
#   configs/training/anchordiff_v26_FULL_DATA_local.yaml                     (from Round-25)
#
# Usage:
#   bash scripts/stage_b_generator/run_round26_v27_anchor_diag.sh
#
# Skip-pair via env (resume after partial failure):
#   ROUND26_DIAG_RESUME_FROM=pair_2 bash <this script>   # skip PAIR-1
#   ROUND26_DIAG_RESUME_FROM=pack    bash <this script>   # only re-pack

set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="runs/round26_v27_anchor_diag"
mkdir -p "${LOG_DIR}"

V27_CFG_LOCAL="configs/training/anchordiff_v27_stage2_anchoraware_FULL_DATA_local.yaml"
V26_CFG_LOCAL="configs/training/anchordiff_v26_FULL_DATA_local.yaml"
V27_RUN_DIR="runs/training/stageB_anchordiff_v27_stage2_anchoraware_FULL_DATA"
V27_FINAL="${V27_RUN_DIR}/final.pt"
V27_BEST="${V27_RUN_DIR}/best_val.pt"
R23_CKPT="runs/training/stageB_anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA/final.pt"
EVAL_SUBSET="analyses/round25_multimodal_eval_subset.json"

OUT_FINAL="analyses/round26_v27_anchor_diag_final"
OUT_BEST="analyses/round26_v27_anchor_diag_best_val"
OUT_R23="analyses/round26_r23_anchor_diag_48clips"

RESUME_FROM="${ROUND26_DIAG_RESUME_FROM:-}"

_should_skip() {
    local stages=(pair_1 pair_2 pack)
    [[ -z "${RESUME_FROM}" ]] && return 1
    local target_idx=-1 current_idx=-1 i
    for ((i=0; i<${#stages[@]}; i++)); do
        [[ "${stages[i]}" == "$1" ]] && current_idx=$i
        [[ "${stages[i]}" == "${RESUME_FROM}" ]] && target_idx=$i
    done
    [[ $target_idx -lt 0 ]] && { echo "WARN: unknown ROUND26_DIAG_RESUME_FROM=${RESUME_FROM}"; return 1; }
    [[ $current_idx -lt $target_idx ]]
}

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

# ---------- preflight ----------
for F in "${V27_CFG_LOCAL}" "${V26_CFG_LOCAL}" "${V27_FINAL}" "${V27_BEST}" "${R23_CKPT}" "${EVAL_SUBSET}"; do
    if [[ ! -e "${F}" ]]; then
        echo "ERROR: missing prerequisite: ${F}"
        exit 1
    fi
done

# ============================================================
# PAIR-1: v27 final (cuda:0) || v27 best_val (cuda:1)
# ============================================================
if _should_skip pair_1; then
    echo "[SKIP] PAIR-1 (ROUND26_DIAG_RESUME_FROM=${RESUME_FROM})"
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PAIR-1: v27 final (cuda:0) || v27 best_val (cuda:1)"
    echo "================================================================"
    run_step_bg "anchor_diag_v27_final" 0 \
        conda run --no-capture-output -n piano python -u \
            scripts/stage_b_generator/anchor_realization_diagnostic.py \
            --config "${V27_CFG_LOCAL}" --ckpt "${V27_FINAL}" \
            --selection-json "${EVAL_SUBSET}" \
            --output-dir "${OUT_FINAL}" \
            --bucket val --cfg-scale 1.0 --seed 42 &
    PID_FINAL=$!

    run_step_bg "anchor_diag_v27_best_val" 1 \
        conda run --no-capture-output -n piano python -u \
            scripts/stage_b_generator/anchor_realization_diagnostic.py \
            --config "${V27_CFG_LOCAL}" --ckpt "${V27_BEST}" \
            --selection-json "${EVAL_SUBSET}" \
            --output-dir "${OUT_BEST}" \
            --bucket val --cfg-scale 1.0 --seed 42 &
    PID_BEST=$!

    echo "    PID final=${PID_FINAL}  best_val=${PID_BEST}"
    echo "    follow: tail -f ${LOG_DIR}/anchor_diag_v27_final.log ${LOG_DIR}/anchor_diag_v27_best_val.log"

    EX_FINAL=0; EX_BEST=0
    wait $PID_FINAL || EX_FINAL=$?
    wait $PID_BEST  || EX_BEST=$?
    if [[ $EX_FINAL -ne 0 || $EX_BEST -ne 0 ]]; then
        echo "WARN: PAIR-1 failures (final=${EX_FINAL} best_val=${EX_BEST}); continuing to PAIR-2 anyway."
    fi
    echo "[$(date '+%F %T')] PAIR-1 DONE (final=${EX_FINAL} best_val=${EX_BEST})"
fi

# ============================================================
# PAIR-2: R23 baseline on the same 48-clip subset (cuda:0 solo)
# ============================================================
if _should_skip pair_2; then
    echo "[SKIP] PAIR-2 (ROUND26_DIAG_RESUME_FROM=${RESUME_FROM})"
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PAIR-2: R23 baseline (cuda:0, cuda:1 idle)"
    echo "================================================================"
    EX_R23=0
    run_step "anchor_diag_r23_48clips" \
        env CUDA_VISIBLE_DEVICES=0 \
        conda run --no-capture-output -n piano python -u \
            scripts/stage_b_generator/anchor_realization_diagnostic.py \
            --config "${V26_CFG_LOCAL}" --ckpt "${R23_CKPT}" \
            --selection-json "${EVAL_SUBSET}" \
            --output-dir "${OUT_R23}" \
            --bucket val --cfg-scale 1.0 --seed 42 \
        || EX_R23=$?
    if [[ $EX_R23 -ne 0 ]]; then
        echo "WARN: PAIR-2 failed (R23=${EX_R23})."
    fi
fi

# ============================================================
# PACK: tar all anchor diag outputs into a single tarball
# ============================================================
echo
echo "================================================================"
echo "[$(date '+%F %T')] PACK results"
echo "================================================================"
run_step "pack" \
    bash scripts/stage_b_generator/round26_v27_anchor_diag_pack.sh

echo
echo "================================================================"
echo "Round-26 v27 per-part anchor diagnostic complete."
echo "Outputs:"
echo "  v27 final per-part:    ${OUT_FINAL}/anchor_summary.md"
echo "  v27 best_val per-part: ${OUT_BEST}/anchor_summary.md"
echo "  R23 baseline per-part: ${OUT_R23}/anchor_summary.md"
echo "  Stage logs:            ${LOG_DIR}/*.log"
echo "  Tarball:               round26_v27_anchor_diag_*.tar.gz (see PACK output above)"
echo "================================================================"
