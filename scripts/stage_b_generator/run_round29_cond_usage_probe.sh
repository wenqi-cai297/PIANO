#!/usr/bin/env bash
# Round-29 cond-usage probe launcher (Phase 0 — mandatory pre-PB).
#
# Per Codex review §3 of analyses/2026-05-29_round29_cond_injection_prior_codex_review_for_claude_code.md:
# before any architectural ablation (PB1/PB2), we must directly measure
# whether each R29 cond family (coarse_extra / interaction / support /
# body_refine) is actually consumed by the existing A1 / R0 / G1 ckpts,
# rather than inferring from cross-run paired bootstrap.
#
# This launcher runs round29_cond_usage_probe.py against each of the
# three reference ckpts on the val 48-clip balanced subset, then packs
# the per-ckpt outputs into a tarball for scp back to local.
#
# No training. Sampling only. Wall-clock estimate on 2x 5080:
#   N_clips=48, baseline + 5 perturbations × (avg 3 active families) per ckpt
#   = 48 × (1 + 5 × 3) = 768 samples per ckpt
#   ≈ 25-35 min per ckpt → 3 ckpts ≈ 75-105 min total (dual-GPU pairwise)
#   or ≈ 2.5-3.5 h single-GPU sequential.
#
# Default ckpt set:
#   A1 r29_ns_a1_c41_s4_g1                 (cond probed: coarse_extra, support)
#   R0 r29_ft_r0_clean_a3_baseline         (cond probed: all 4 — C/I/S/B)
#   G1 r29_nb_g1_phasefree_gait_fixed      (cond probed: all 4)
#
# Prerequisites on the Linux server:
#   1. git pull (this script + the probe python)
#   2. ffmpeg NOT required (no rendering)
#   3. ckpts present at:
#        runs/training/stageB_anchordiff_r29_ns_a1_c41_s4_g1/final.pt
#        runs/training/stageB_anchordiff_r29_ft_r0_clean_a3_baseline/final.pt
#        runs/training/stageB_anchordiff_r29_nb_g1_phasefree_gait_fixed/final.pt
#   4. val 48-clip selection JSON:
#        analyses/round29_val_diag_indices_48_balanced.json
#   5. configs regenerated with server data root (DATASETS_ROOT or the
#      launcher will offer commands to regenerate)
#
# Usage:
#   bash scripts/stage_b_generator/run_round29_cond_usage_probe.sh
#
#   # Probe a subset only:
#   ROUND29_CU_VARIANTS="r29_ns_a1_c41_s4_g1" \
#     bash scripts/stage_b_generator/run_round29_cond_usage_probe.sh
#
#   # Different bucket:
#   ROUND29_CU_BUCKET=train SEL_JSON=analyses/round27_tier0_train_indices_48_balanced.json \
#     bash scripts/stage_b_generator/run_round29_cond_usage_probe.sh
#
#   # Single-GPU sequential fallback:
#   ROUND29_CU_SINGLE_GPU=1 bash scripts/stage_b_generator/run_round29_cond_usage_probe.sh
#
#   # Custom perturbation set:
#   ROUND29_CU_PERTURBATIONS="baseline,zero,time_shuffle" \
#     bash scripts/stage_b_generator/run_round29_cond_usage_probe.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="runs/round29_cond_usage_probe"
mkdir -p "${LOG_DIR}"

# ---------- env overrides ----------
SEL_JSON="${SEL_JSON:-analyses/round29_val_diag_indices_48_balanced.json}"
BUCKET="${ROUND29_CU_BUCKET:-val}"
SINGLE_GPU="${ROUND29_CU_SINGLE_GPU:-0}"
CKPT_NAME="${ROUND29_CU_CKPT_NAME:-final.pt}"
PERTURBATIONS="${ROUND29_CU_PERTURBATIONS:-baseline,zero,time_shuffle,batch_shuffle,scale_0.5,scale_2.0}"

DEFAULT_VARIANTS=(
    r29_ns_a1_c41_s4_g1
    r29_ft_r0_clean_a3_baseline
    r29_nb_g1_phasefree_gait_fixed
)
if [[ -n "${ROUND29_CU_VARIANTS:-}" ]]; then
    # shellcheck disable=SC2206
    VARIANTS=(${ROUND29_CU_VARIANTS})
else
    VARIANTS=("${DEFAULT_VARIANTS[@]}")
fi

# ---------- preflight ----------
PREFLIGHT_FAIL=0

# Selection JSON.
if [[ ! -e "${SEL_JSON}" ]]; then
    echo "[CU PREFLIGHT FAIL] missing selection JSON: ${SEL_JSON}"
    if [[ "${SEL_JSON}" == *round29_val_diag_indices_48_balanced.json ]]; then
        echo "    -> rebuild via:"
        echo "       python scripts/stage_b_generator/round29_build_val_diag_subset.py \\"
        echo "           --config configs/training/anchordiff_r29_ns_a1_c41_s4_g1.yaml"
    fi
    PREFLIGHT_FAIL=1
fi

# Configs + ckpts.
for V in "${VARIANTS[@]}"; do
    CFG="configs/training/anchordiff_${V}.yaml"
    CKPT="runs/training/stageB_anchordiff_${V}/${CKPT_NAME}"
    if [[ ! -e "${CFG}" ]]; then
        echo "[CU PREFLIGHT FAIL] missing config: ${CFG}"
        if [[ "${V}" == r29_ns_* ]]; then
            echo "    -> regenerate via:"
            echo "       export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4"
            echo "       python scripts/stage_b_generator/round29_make_next_step_ablation_configs.py"
        elif [[ "${V}" == r29_nb_* ]]; then
            echo "    -> regenerate via:"
            echo "       python scripts/stage_b_generator/round29_make_next_ablation_configs.py"
        elif [[ "${V}" == r29_ft_* ]]; then
            echo "    -> regenerate via:"
            echo "       python scripts/stage_b_generator/round29_make_failure_targeted_ablation_configs.py"
        fi
        PREFLIGHT_FAIL=1
    fi
    if [[ ! -e "${CKPT}" ]]; then
        echo "[CU PREFLIGHT FAIL] missing ckpt: ${CKPT}"
        PREFLIGHT_FAIL=1
    fi
done

if [[ ${PREFLIGHT_FAIL} -ne 0 ]]; then
    echo "[CU] FATAL preflight failures."
    exit 1
fi

echo "[CU] variants:      ${VARIANTS[*]}"
echo "[CU] bucket:        ${BUCKET}"
echo "[CU] selection:     ${SEL_JSON}"
echo "[CU] perturbations: ${PERTURBATIONS}"
echo "[CU] single_gpu:    ${SINGLE_GPU}"
echo "[CU] ckpt_name:     ${CKPT_NAME}"

# ---------- helpers ----------
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
        env CUDA_VISIBLE_DEVICES="${GPU}" \
            PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$@" 2>&1
        local T1
        T1=$(date +%s)
        echo "[$(date '+%F %T')] BG-DONE ${NAME} in $((T1-T0))s"
    } > "${LOG}" 2>&1
}

run_step_fg() {
    local NAME="$1"; shift
    local GPU="$1"; shift
    local LOG="${LOG_DIR}/${NAME}.log"
    local T0
    T0=$(date +%s)
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] START ${NAME} on cuda:${GPU}"
    echo "    log: ${LOG}"
    echo "================================================================"
    env CUDA_VISIBLE_DEVICES="${GPU}" \
        PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$@" 2>&1 | tee "${LOG}"
    local T1
    T1=$(date +%s)
    echo "[$(date '+%F %T')] DONE ${NAME} in $((T1-T0))s"
}

probe_one() {
    local NAME="$1"; local GPU="$2"; local CFG="$3"; local CKPT="$4"; local OUT="$5"; local MODE="$6"
    local RUN_FN="run_step_fg"
    if [[ "${MODE}" == "bg" ]]; then RUN_FN="run_step_bg"; fi
    mkdir -p "${OUT}"
    "${RUN_FN}" "probe_${NAME}" "${GPU}" \
        conda run --no-capture-output -n piano python -u \
            scripts/stage_b_generator/round29_cond_usage_probe.py \
            --config "${CFG}" --ckpt "${CKPT}" \
            --selection-json "${SEL_JSON}" \
            --output-dir "${OUT}" \
            --bucket "${BUCKET}" \
            --variant-id "${NAME}" \
            --perturbations "${PERTURBATIONS}" \
            --cfg-scale 1.0 --seed 42
}

OUT_ROOT="analyses/round29_cond_usage_probe"
mkdir -p "${OUT_ROOT}"

# ---------- run ----------
if [[ "${SINGLE_GPU}" == "1" ]]; then
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] SINGLE-GPU RUN (cuda:0, sequential ${#VARIANTS[@]} ckpts)"
    echo "================================================================"
    for V in "${VARIANTS[@]}"; do
        CFG="configs/training/anchordiff_${V}.yaml"
        CKPT="runs/training/stageB_anchordiff_${V}/${CKPT_NAME}"
        OUT="${OUT_ROOT}/${V}_${BUCKET}"
        probe_one "${V}" 0 "${CFG}" "${CKPT}" "${OUT}" fg
    done
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] DUAL-GPU RUN: pairwise (cuda:0 || cuda:1)"
    echo "================================================================"
    i=0
    PIDS=()
    NAMES=()
    for V in "${VARIANTS[@]}"; do
        CFG="configs/training/anchordiff_${V}.yaml"
        CKPT="runs/training/stageB_anchordiff_${V}/${CKPT_NAME}"
        OUT="${OUT_ROOT}/${V}_${BUCKET}"
        GPU=$((i % 2))
        probe_one "${V}" "${GPU}" "${CFG}" "${CKPT}" "${OUT}" bg &
        PIDS+=($!)
        NAMES+=("${V}")
        i=$((i + 1))
        if (( i % 2 == 0 )); then
            for j in 0 1; do
                wait "${PIDS[j]}" || echo "  ${NAMES[j]} exited non-zero"
            done
            PIDS=()
            NAMES=()
        fi
    done
    for k in "${!PIDS[@]}"; do
        wait "${PIDS[k]}" || echo "  ${NAMES[k]} exited non-zero"
    done
fi

# ---------- pack ----------
STAMP=$(date +%Y%m%d_%H%M%S)
TARBALL="round29_cond_usage_probe_${STAMP}.tar.gz"
echo
echo "================================================================"
echo "[$(date '+%F %T')] PACK -> ${TARBALL}"
echo "================================================================"
tar -czf "${TARBALL}" -C analyses "$(basename "${OUT_ROOT}")"
SIZE=$(du -h "${TARBALL}" | cut -f1)
echo "wrote ${TARBALL}  (${SIZE})"

echo
echo "================================================================"
echo "Round-29 cond-usage probe complete."
echo "Outputs:"
for V in "${VARIANTS[@]}"; do
    echo "  ${OUT_ROOT}/${V}_${BUCKET}/{cond_usage_stats.json, cond_usage_summary.md}"
done
echo "Logs:    ${LOG_DIR}/probe_*.log"
echo "Tarball: ${TARBALL}"
echo
echo "scp back:  scp <server>:$(pwd)/${TARBALL} ."
echo "Then on local:  tar -xzf ${TARBALL} -C analyses/"
echo "================================================================"
