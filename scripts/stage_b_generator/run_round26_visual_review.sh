#!/usr/bin/env bash
# Round-26 visual review launcher — DUAL-GPU.
#
# Renders 24 clips × 2 ckpts (R23 baseline + v27 final) using
# scripts/stage_b_generator/render_round24_visual_review.py with the
# 24-clip selection JSON produced by round26_air_grab_analysis.py.
#
# The 24 clips are split into 4 categories (per Codex Round-26 review §5):
#   - 8 fixed_by_v27        : v27 reduced air-grab vs R23
#   - 8 still_bad_hand      : v27 hands still have air-grab > tau
#   - 4 v27_regressed       : v27 made air-grab worse than R23
#   - 4 d3_sampled_failure  : largest D3 sampled-coarse gap
#
# Wall-clock on 2× A6000:
#   PAIR (R23 cuda:0 || v27_final cuda:1)   ~12-15 min  (24 clips × ~30s/clip × 2 renders/clip / 2 GPU)
#   PACK                                     ~30 s
#   TOTAL                                    ~15 min
#
# Prerequisites:
#   git pull (must include the round26 air-grab analysis script)
#   analyses/round26_visual_review_selection.json (24 clips, produced
#     locally by `python scripts/stage_b_generator/round26_air_grab_analysis.py`)
#   configs/training/anchordiff_v27_stage2_anchoraware_FULL_DATA_local.yaml
#   configs/training/anchordiff_v26_FULL_DATA_local.yaml
#   v27 final ckpt + R23 no-plan ckpt
#
# Usage:
#   bash scripts/stage_b_generator/run_round26_visual_review.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="runs/round26_visual_review"
mkdir -p "${LOG_DIR}"

V27_CFG_LOCAL="configs/training/anchordiff_v27_stage2_anchoraware_FULL_DATA_local.yaml"
V26_CFG_LOCAL="configs/training/anchordiff_v26_FULL_DATA_local.yaml"
V27_FINAL="runs/training/stageB_anchordiff_v27_stage2_anchoraware_FULL_DATA/final.pt"
R23_CKPT="runs/training/stageB_anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA/final.pt"
SEL_JSON="analyses/round26_visual_review_selection.json"

OUT_R23="analyses/round26_visual_review/r23_baseline"
OUT_V27="analyses/round26_visual_review/v27_final"

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

# ---------- preflight ----------
for F in "${V27_CFG_LOCAL}" "${V26_CFG_LOCAL}" "${V27_FINAL}" "${R23_CKPT}" "${SEL_JSON}"; do
    if [[ ! -e "${F}" ]]; then
        echo "ERROR: missing prerequisite: ${F}"
        exit 1
    fi
done

# How many clips are in the selection?
N_CLIPS=$(python -c "import json; print(json.load(open('${SEL_JSON}'))['n_clips'])")
echo "Selection: ${N_CLIPS} clips from ${SEL_JSON}"

mkdir -p "${OUT_R23}" "${OUT_V27}"

echo
echo "================================================================"
echo "[$(date '+%F %T')] DUAL-GPU RENDER: R23 (cuda:0) || v27 final (cuda:1)"
echo "================================================================"

run_step_bg "render_r23_baseline" 0 \
    conda run --no-capture-output -n piano python -u \
        scripts/stage_b_generator/render_round24_visual_review.py \
        --config "${V26_CFG_LOCAL}" --ckpt "${R23_CKPT}" \
        --selection-json "${SEL_JSON}" \
        --output-dir "${OUT_R23}" \
        --bucket val --n-clips "${N_CLIPS}" --cfg-scale 1.0 --seed 42 &
PID_R23=$!

run_step_bg "render_v27_final" 1 \
    conda run --no-capture-output -n piano python -u \
        scripts/stage_b_generator/render_round24_visual_review.py \
        --config "${V27_CFG_LOCAL}" --ckpt "${V27_FINAL}" \
        --selection-json "${SEL_JSON}" \
        --output-dir "${OUT_V27}" \
        --bucket val --n-clips "${N_CLIPS}" --cfg-scale 1.0 --seed 42 &
PID_V27=$!

echo "    PID r23=${PID_R23}  v27=${PID_V27}"
echo "    follow: tail -f ${LOG_DIR}/render_r23_baseline.log ${LOG_DIR}/render_v27_final.log"

EX_R23=0; EX_V27=0
wait $PID_R23 || EX_R23=$?
wait $PID_V27 || EX_V27=$?
if [[ $EX_R23 -ne 0 || $EX_V27 -ne 0 ]]; then
    echo "WARN: render failures (r23=${EX_R23} v27=${EX_V27}); continuing to PACK anyway."
fi
echo "[$(date '+%F %T')] RENDER DONE (r23=${EX_R23} v27=${EX_V27})"

# ============================================================
# PACK
# ============================================================
echo
echo "================================================================"
echo "[$(date '+%F %T')] PACK results"
echo "================================================================"
bash scripts/stage_b_generator/round26_visual_review_pack.sh

echo
echo "================================================================"
echo "Round-26 visual review complete."
echo "Outputs:"
echo "  R23 videos:      ${OUT_R23}/clip*_{gt,pred}.mp4  + summary.md"
echo "  v27 videos:      ${OUT_V27}/clip*_{gt,pred}.mp4  + summary.md"
echo "  Selection JSON:  ${SEL_JSON}"
echo "  Stage logs:      ${LOG_DIR}/*.log"
echo "  Tarball:         round26_visual_review_*.tar.gz (see PACK output)"
echo "================================================================"
