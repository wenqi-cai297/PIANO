#!/usr/bin/env bash
# Round-26 temporal diagnostics launcher: sustained-contact + gait.
#
# Two new diagnostic scripts beyond Round-24 per-anchor measurement:
#
#   round26_sustained_contact_diag.py
#     For each contiguous contact segment in pseudo-labels (>= 5 frames),
#     measure whether predicted limb stays close to object surface
#     throughout the segment vs only at single anchor frames. Captures
#     the "hand follows box partway then lets go" failure mode.
#
#   round26_gait_diag.py
#     Detect walking segments (root horizontal speed > threshold sustained
#     >= 20 frames). Within each, measure foot alternation (both-stance,
#     both-swing, L-R height anti-phase correlation, step period). Captures
#     the "feet don't alternate properly" failure mode.
#
# Runs on the same 48-clip multimodal eval subset used by D2/D3/anchor-diag
# for direct comparability. Computes both metrics on three sources:
#
#   cuda:0  sustained_contact + gait  on  v27 final ckpt
#   cuda:1  sustained_contact + gait  on  R23 baseline ckpt
#   CPU/short cuda:0  sustained_contact + gait  on  GT motion (sanity baseline;
#                     drift should be ~0, tracking_fraction ~1.0, gait should
#                     match human walking; if GT itself looks bad, the metric
#                     threshold needs adjustment)
#
# Wall-clock on 2× A6000:
#   PAIR-A (sustained_contact v27 cuda:0 || R23 cuda:1)   ~30 min
#   PAIR-B (gait v27 cuda:0 || R23 cuda:1)                 ~30 min
#   GT REF (sustained + gait on GT motion, no sampling)    ~5 min
#   PACK                                                    ~30 s
#   TOTAL                                                  ~65 min

set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="runs/round26_temporal_diag"
mkdir -p "${LOG_DIR}"

V27_CFG_LOCAL="configs/training/anchordiff_v27_stage2_anchoraware_FULL_DATA_local.yaml"
V26_CFG_LOCAL="configs/training/anchordiff_v26_FULL_DATA_local.yaml"
V27_FINAL="runs/training/stageB_anchordiff_v27_stage2_anchoraware_FULL_DATA/final.pt"
R23_CKPT="runs/training/stageB_anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA/final.pt"
SEL_JSON="analyses/round25_multimodal_eval_subset.json"

SUS_V27="analyses/round26_sustained_contact_v27_final"
SUS_R23="analyses/round26_sustained_contact_r23"
SUS_GT="analyses/round26_sustained_contact_gt_reference"

GAIT_V27="analyses/round26_gait_v27_final"
GAIT_R23="analyses/round26_gait_r23"
GAIT_GT="analyses/round26_gait_gt_reference"

RESUME_FROM="${ROUND26_TEMPORAL_RESUME_FROM:-}"

_should_skip() {
    local stages=(pair_a pair_b gt_ref pack)
    [[ -z "${RESUME_FROM}" ]] && return 1
    local target=-1 cur=-1 i
    for ((i=0; i<${#stages[@]}; i++)); do
        [[ "${stages[i]}" == "$1" ]] && cur=$i
        [[ "${stages[i]}" == "${RESUME_FROM}" ]] && target=$i
    done
    [[ $target -lt 0 ]] && { echo "WARN: unknown ROUND26_TEMPORAL_RESUME_FROM=${RESUME_FROM}"; return 1; }
    [[ $cur -lt $target ]]
}

run_step() {
    local NAME="$1"; shift
    local LOG="${LOG_DIR}/${NAME}.log"
    local T0; T0=$(date +%s)
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] START ${NAME}"
    echo "    log: ${LOG}"
    echo "================================================================"
    PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$@" 2>&1 | tee "${LOG}"
    local T1; T1=$(date +%s)
    echo "[$(date '+%F %T')] DONE ${NAME} in $((T1-T0))s"
}

run_step_bg() {
    local NAME="$1"; shift
    local GPU="$1"; shift
    local LOG="${LOG_DIR}/${NAME}.log"
    local T0; T0=$(date +%s)
    {
        echo "[$(date '+%F %T')] BG-START ${NAME} on cuda:${GPU}"
        CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$@" 2>&1
        local T1; T1=$(date +%s)
        echo "[$(date '+%F %T')] BG-DONE ${NAME} in $((T1-T0))s"
    } > "${LOG}" 2>&1
}

# ---------- preflight ----------
for F in "${V27_CFG_LOCAL}" "${V26_CFG_LOCAL}" "${V27_FINAL}" "${R23_CKPT}" "${SEL_JSON}"; do
    [[ -e "${F}" ]] || { echo "ERROR: missing prerequisite: ${F}"; exit 1; }
done

# ============================================================
# PAIR-A: sustained contact (v27 cuda:0 || R23 cuda:1)
# ============================================================
if _should_skip pair_a; then
    echo "[SKIP] PAIR-A"
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PAIR-A: sustained_contact v27 (cuda:0) || R23 (cuda:1)"
    echo "================================================================"
    run_step_bg "sustained_contact_v27_final" 0 \
        conda run --no-capture-output -n piano python -u \
            scripts/stage_b_generator/round26_sustained_contact_diag.py \
            --config "${V27_CFG_LOCAL}" --ckpt "${V27_FINAL}" \
            --selection-json "${SEL_JSON}" \
            --output-dir "${SUS_V27}" \
            --bucket val --cfg-scale 1.0 --seed 42 &
    PID_A1=$!

    run_step_bg "sustained_contact_r23" 1 \
        conda run --no-capture-output -n piano python -u \
            scripts/stage_b_generator/round26_sustained_contact_diag.py \
            --config "${V26_CFG_LOCAL}" --ckpt "${R23_CKPT}" \
            --selection-json "${SEL_JSON}" \
            --output-dir "${SUS_R23}" \
            --bucket val --cfg-scale 1.0 --seed 42 &
    PID_A2=$!

    echo "    PID v27=${PID_A1}  r23=${PID_A2}"
    echo "    follow: tail -f ${LOG_DIR}/sustained_contact_v27_final.log ${LOG_DIR}/sustained_contact_r23.log"
    EX_A1=0; EX_A2=0
    wait $PID_A1 || EX_A1=$?
    wait $PID_A2 || EX_A2=$?
    if [[ $EX_A1 -ne 0 || $EX_A2 -ne 0 ]]; then
        echo "WARN: PAIR-A failures (v27=${EX_A1} r23=${EX_A2}); continuing"
    fi
    echo "[$(date '+%F %T')] PAIR-A DONE"
fi

# ============================================================
# PAIR-B: gait (v27 cuda:0 || R23 cuda:1)
# ============================================================
if _should_skip pair_b; then
    echo "[SKIP] PAIR-B"
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PAIR-B: gait v27 (cuda:0) || R23 (cuda:1)"
    echo "================================================================"
    run_step_bg "gait_v27_final" 0 \
        conda run --no-capture-output -n piano python -u \
            scripts/stage_b_generator/round26_gait_diag.py \
            --config "${V27_CFG_LOCAL}" --ckpt "${V27_FINAL}" \
            --selection-json "${SEL_JSON}" \
            --output-dir "${GAIT_V27}" \
            --bucket val --cfg-scale 1.0 --seed 42 &
    PID_B1=$!

    run_step_bg "gait_r23" 1 \
        conda run --no-capture-output -n piano python -u \
            scripts/stage_b_generator/round26_gait_diag.py \
            --config "${V26_CFG_LOCAL}" --ckpt "${R23_CKPT}" \
            --selection-json "${SEL_JSON}" \
            --output-dir "${GAIT_R23}" \
            --bucket val --cfg-scale 1.0 --seed 42 &
    PID_B2=$!

    echo "    PID v27=${PID_B1}  r23=${PID_B2}"
    echo "    follow: tail -f ${LOG_DIR}/gait_v27_final.log ${LOG_DIR}/gait_r23.log"
    EX_B1=0; EX_B2=0
    wait $PID_B1 || EX_B1=$?
    wait $PID_B2 || EX_B2=$?
    if [[ $EX_B1 -ne 0 || $EX_B2 -ne 0 ]]; then
        echo "WARN: PAIR-B failures (v27=${EX_B1} r23=${EX_B2}); continuing"
    fi
    echo "[$(date '+%F %T')] PAIR-B DONE"
fi

# ============================================================
# GT REF: sustained + gait on GT motion (fast, no sampling)
# ============================================================
if _should_skip gt_ref; then
    echo "[SKIP] GT_REF"
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] GT REFERENCE: sustained + gait on GT (cuda:0 sequential)"
    echo "================================================================"
    EX_GT_SUS=0
    run_step "sustained_contact_gt_reference" \
        env CUDA_VISIBLE_DEVICES=0 \
        conda run --no-capture-output -n piano python -u \
            scripts/stage_b_generator/round26_sustained_contact_diag.py \
            --config "${V27_CFG_LOCAL}" --ckpt "${V27_FINAL}" \
            --selection-json "${SEL_JSON}" \
            --output-dir "${SUS_GT}" \
            --bucket val --use-gt-as-pred \
        || EX_GT_SUS=$?

    EX_GT_GAIT=0
    run_step "gait_gt_reference" \
        env CUDA_VISIBLE_DEVICES=0 \
        conda run --no-capture-output -n piano python -u \
            scripts/stage_b_generator/round26_gait_diag.py \
            --config "${V27_CFG_LOCAL}" --ckpt "${V27_FINAL}" \
            --selection-json "${SEL_JSON}" \
            --output-dir "${GAIT_GT}" \
            --bucket val --use-gt-as-pred \
        || EX_GT_GAIT=$?

    if [[ $EX_GT_SUS -ne 0 || $EX_GT_GAIT -ne 0 ]]; then
        echo "WARN: GT_REF failures (sus=${EX_GT_SUS} gait=${EX_GT_GAIT}); continuing"
    fi
fi

# ============================================================
# PACK
# ============================================================
echo
echo "================================================================"
echo "[$(date '+%F %T')] PACK results"
echo "================================================================"
bash scripts/stage_b_generator/round26_temporal_diag_pack.sh

echo
echo "================================================================"
echo "Round-26 temporal diagnostics complete."
echo "Outputs:"
echo "  Sustained contact: ${SUS_V27}/  ${SUS_R23}/  ${SUS_GT}/"
echo "  Gait:              ${GAIT_V27}/ ${GAIT_R23}/ ${GAIT_GT}/"
echo "  Stage logs:        ${LOG_DIR}/*.log"
echo "  Tarball:           round26_temporal_diag_*.tar.gz"
echo "================================================================"
