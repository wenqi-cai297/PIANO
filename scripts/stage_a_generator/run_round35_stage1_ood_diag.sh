#!/usr/bin/env bash
# Round-35 Stage-1 coarse OOD diagnostics.
#
# This is a diagnostic launcher, not a training launcher. It answers:
#   1. How different is generated Stage-1 stage1_coarse from oracle?
#   2. How does Stage-1.5 + frozen PB1 drift change as we interpolate
#      oracle -> generated stage1_coarse?
#
# Pipeline:
#   audit generated-vs-oracle stage1_coarse
#   build alpha substitute caches:
#       mixed = oracle + alpha * (generated - oracle)
#   for each alpha, reuse run_round32_stage1p5_downstream_diag.sh with
#       ROUND32_DS_UPSTREAM_DIR=<alpha-cache-root>
#
# Usage:
#   export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
#   bash scripts/stage_a_generator/run_round35_stage1_ood_diag.sh
#
# Common overrides:
#   ROUND35_STAGE1P5_VARIANT=stage1p5_r34v2_a_lambda0p005
#   ROUND35_STAGE1P5_CFG=configs/training/${ROUND35_STAGE1P5_VARIANT}.yaml
#   ROUND35_STAGE1P5_CKPT=runs/training/${ROUND35_STAGE1P5_VARIANT}/final.pt
#   ROUND35_GENERATED_STAGE1_DIR=analyses/round31_stage1_substitute_conds_v8_stage1_v8_v6_full_f1
#   ROUND35_ALPHAS="0.00,0.25,0.50,0.75,1.00"
#   ROUND35_BUCKETS="val"

set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
SKIP_AUDIT=0
SKIP_BUILD=0
SKIP_DIAG=0
ONLY_ALPHA=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=1; shift ;;
        --skip-audit)   SKIP_AUDIT=1; shift ;;
        --skip-build)   SKIP_BUILD=1; shift ;;
        --skip-diag)    SKIP_DIAG=1; shift ;;
        --only-alpha)   ONLY_ALPHA="$2"; shift 2 ;;
        -h|--help)      sed -n '1,45p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python";
    elif command -v python3 >/dev/null 2>&1; then PY="python3";
    else echo "[R35] FATAL: no python found" >&2; exit 127; fi
fi

STAGE1P5_VARIANT="${ROUND35_STAGE1P5_VARIANT:-stage1p5_r34v2_a_lambda0p005}"
STAGE1P5_CFG="${ROUND35_STAGE1P5_CFG:-configs/training/${STAGE1P5_VARIANT}.yaml}"
STAGE1P5_CKPT="${ROUND35_STAGE1P5_CKPT:-runs/training/${STAGE1P5_VARIANT}/final.pt}"
PB1_CKPT="${ROUND35_PB1_CKPT:-runs/training/stageB_anchordiff_r29_pb_a1_adaln_s4/final.pt}"
GENERATED_STAGE1_DIR="${ROUND35_GENERATED_STAGE1_DIR:-analyses/round31_stage1_substitute_conds_v8_stage1_v8_v6_full_f1}"
BUCKETS_STR="${ROUND35_BUCKETS:-val}"
ALPHAS_STR="${ROUND35_ALPHAS:-0.00,0.25,0.50,0.75,1.00}"
SELECTION_VAL="${ROUND35_SELECTION_VAL:-analyses/round29_val_diag_indices_48_balanced.json}"
SELECTION_TRAIN="${ROUND35_SELECTION_TRAIN:-analyses/round27_tier0_train_indices_48_balanced.json}"
OUT_ROOT="${ROUND35_OUT_ROOT:-analyses/round35_stage1_coarse_ood_${STAGE1P5_VARIANT}}"
LOG_DIR="${ROUND35_LOG_DIR:-runs/round35_stage1_ood_${STAGE1P5_VARIANT}}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

echo "[R35] STAGE1P5_VARIANT=${STAGE1P5_VARIANT}"
echo "[R35] STAGE1P5_CFG=${STAGE1P5_CFG}"
echo "[R35] STAGE1P5_CKPT=${STAGE1P5_CKPT}"
echo "[R35] PB1_CKPT=${PB1_CKPT}"
echo "[R35] GENERATED_STAGE1_DIR=${GENERATED_STAGE1_DIR}"
echo "[R35] BUCKETS=${BUCKETS_STR}"
echo "[R35] ALPHAS=${ALPHAS_STR}"
echo "[R35] OUT_ROOT=${OUT_ROOT}"

preflight_fail=0
for p in "${STAGE1P5_CFG}" "${GENERATED_STAGE1_DIR}" \
         scripts/stage_a_generator/round35_stage1_coarse_ood_audit.py \
         scripts/stage_a_generator/build_stage1_coarse_residual_variants.py \
         scripts/stage_a_generator/run_round32_stage1p5_downstream_diag.sh; do
    [[ ! -e "${p}" ]] && { echo "[R35 PREFLIGHT FAIL] missing: ${p}"; preflight_fail=1; }
done
if [[ ${DRY_RUN} -eq 0 ]]; then
    for p in "${STAGE1P5_CKPT}" "${PB1_CKPT}"; do
        [[ ! -e "${p}" ]] && { echo "[R35 PREFLIGHT FAIL] missing ckpt: ${p}"; preflight_fail=1; }
    done
fi
if [[ ${preflight_fail} -ne 0 ]]; then
    echo "[R35] FATAL preflight failures."
    exit 1
fi

# shellcheck disable=SC2206
BUCKETS=(${BUCKETS_STR})

alpha_tag() {
    "${PY}" - "$1" <<'PY'
import sys
a = float(sys.argv[1])
print(f"alpha{int(round(a * 100)):03d}")
PY
}

for bucket in "${BUCKETS[@]}"; do
    case "${bucket}" in
        val) selection="${SELECTION_VAL}" ;;
        train) selection="${SELECTION_TRAIN}" ;;
        *) echo "[R35] unknown bucket: ${bucket}" >&2; exit 2 ;;
    esac

    if [[ ${SKIP_AUDIT} -eq 0 ]]; then
        AUDIT_MD="${OUT_ROOT}/stage1_coarse_ood_audit_${bucket}.md"
        AUDIT_LOG="${LOG_DIR}/audit_${bucket}.log"
        AUDIT_CMD=("${PY}" -u scripts/stage_a_generator/round35_stage1_coarse_ood_audit.py
            --config "${STAGE1P5_CFG}"
            --generated-dir "${GENERATED_STAGE1_DIR}"
            --selection-json "${selection}"
            --bucket "${bucket}"
            --out-md "${AUDIT_MD}")
        echo
        echo "================================================================"
        echo "[$(date '+%F %T')] R35 AUDIT ${bucket}"
        echo "================================================================"
        if [[ ${DRY_RUN} -eq 1 ]]; then
            echo "[R35 DRY-RUN] ${AUDIT_CMD[*]}"
        else
            "${AUDIT_CMD[@]}" 2>&1 | tee "${AUDIT_LOG}"
        fi
    fi

    if [[ ${SKIP_BUILD} -eq 0 ]]; then
        BUILD_LOG="${LOG_DIR}/build_alpha_${bucket}.log"
        BUILD_CMD=("${PY}" -u scripts/stage_a_generator/build_stage1_coarse_residual_variants.py
            --config "${STAGE1P5_CFG}"
            --generated-dir "${GENERATED_STAGE1_DIR}"
            --selection-json "${selection}"
            --bucket "${bucket}"
            --out-root "${OUT_ROOT}/alpha_caches"
            --alphas "${ALPHAS_STR}")
        echo
        echo "================================================================"
        echo "[$(date '+%F %T')] R35 BUILD ALPHA CACHES ${bucket}"
        echo "================================================================"
        if [[ ${DRY_RUN} -eq 1 ]]; then
            echo "[R35 DRY-RUN] ${BUILD_CMD[*]}"
        else
            "${BUILD_CMD[@]}" 2>&1 | tee "${BUILD_LOG}"
        fi
    fi
done

if [[ ${SKIP_DIAG} -eq 1 ]]; then
    echo "[R35] --skip-diag: alpha caches built, downstream diag skipped."
    exit 0
fi

IFS=',' read -r -a ALPHAS <<< "${ALPHAS_STR}"
for alpha in "${ALPHAS[@]}"; do
    alpha="$(echo "${alpha}" | xargs)"
    [[ -z "${alpha}" ]] && continue
    tag="$(alpha_tag "${alpha}")"
    if [[ -n "${ONLY_ALPHA}" && "${tag}" != "${ONLY_ALPHA}" && "${alpha}" != "${ONLY_ALPHA}" ]]; then
        continue
    fi
    ALPHA_ROOT="${OUT_ROOT}/alpha_caches/${tag}"
    OUT_TAG="_r35_${STAGE1P5_VARIANT}_${tag}"
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] R35 DOWNSTREAM DIAG ${tag} (${alpha})"
    echo "    upstream=${ALPHA_ROOT}"
    echo "================================================================"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "[R35 DRY-RUN] ROUND32_DS_UPSTREAM_DIR=${ALPHA_ROOT} ROUND32_DS_OUT_TAG=${OUT_TAG} ..."
        continue
    fi
    ROUND32_DS_STAGE1P5_CFG="${STAGE1P5_CFG}" \
    ROUND32_DS_STAGE1P5_CKPT="${STAGE1P5_CKPT}" \
    ROUND32_DS_PB1_CKPT="${PB1_CKPT}" \
    ROUND32_DS_BUCKETS="${BUCKETS_STR}" \
    ROUND32_DS_UPSTREAM_DIR="${ALPHA_ROOT}" \
    ROUND32_DS_OUT_TAG="${OUT_TAG}" \
        bash scripts/stage_a_generator/run_round32_stage1p5_downstream_diag.sh \
        2>&1 | tee "${LOG_DIR}/diag_${tag}.log"
done

echo
echo "================================================================"
echo "Round-35 Stage-1 OOD diagnostics complete."
echo "================================================================"
echo "Audit/results root: ${OUT_ROOT}"
echo "Logs: ${LOG_DIR}"

