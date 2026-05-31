#!/usr/bin/env bash
# Round-36 temporal dynamics training + diagnostics launcher.
#
# Default GPU selection is CUDA_VISIBLE_DEVICES=0,2. Override with:
#   ROUND36_CUDA_VISIBLE_DEVICES=0,2
#
# Phases:
#   1) generate R36 configs
#   2) train Stage-1 and Stage-1.5
#   3) sample substitute conds, run temporal dynamics diag, run PB1 downstream diag
#   4) pack small sync artifacts

set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
SKIP_TRAIN=0
SKIP_DIAG=0
SKIP_DOWNSTREAM=0
SKIP_PACK=0
REGEN_CONFIGS=0
ALLOW_PARTIAL="${ROUND36_ALLOW_PARTIAL:-0}"
BUCKETS_STR="${ROUND36_BUCKETS:-val}"
CUDA_DEVICES="${ROUND36_CUDA_VISIBLE_DEVICES:-0,2}"

IFS=',' read -r -a CUDA_LIST <<< "${CUDA_DEVICES}"
NUM_PROCESSES="${ROUND36_NUM_PROCESSES:-${#CUDA_LIST[@]}}"
[[ "${NUM_PROCESSES}" -lt 1 ]] && NUM_PROCESSES=1
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

STAGE1_VID="stage1_r36_v8v6_dynacc"
STAGE1P5_VID="stage1p5_r36_r34v2_a_c41dyn"
STAGE1_CFG="configs/training/${STAGE1_VID}.yaml"
STAGE1P5_CFG="configs/training/${STAGE1P5_VID}.yaml"
STAGE1_CKPT="runs/training/${STAGE1_VID}/final.pt"
STAGE1P5_CKPT="runs/training/${STAGE1P5_VID}/final.pt"
PB1_CKPT="${ROUND36_PB1_CKPT:-runs/training/stageB_anchordiff_r29_pb_a1_adaln_s4/final.pt}"

STAGE1_TAG="_r36_${STAGE1_VID}"
STAGE1P5_ORACLE_TAG="_r36_${STAGE1P5_VID}_oracle"
STAGE1P5_GEN_TAG="_r36_${STAGE1P5_VID}_genstage1"
STAGE1_SUB_ROOT="analyses/round31_stage1_substitute_conds${STAGE1_TAG}"
STAGE1P5_ORACLE_SUB_ROOT="analyses/round32_stage1p5_substitute_conds${STAGE1P5_ORACLE_TAG}"
STAGE1P5_GEN_SUB_ROOT="analyses/round32_stage1p5_substitute_conds${STAGE1P5_GEN_TAG}"
TEMPORAL_ROOT="analyses/round36_temporal_dynamics_diag"
LOG_DIR="runs/round36_dynamics_training_diag"
mkdir -p "${LOG_DIR}"

SELECTION_TRAIN="analyses/round27_tier0_train_indices_48_balanced.json"
SELECTION_VAL="analyses/round29_val_diag_indices_48_balanced.json"

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python";
    elif command -v python3 >/dev/null 2>&1; then PY="python3";
    else echo "[R36] FATAL: neither python nor python3 was found" >&2; exit 127; fi
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)          DRY_RUN=1; shift ;;
        --skip-train)       SKIP_TRAIN=1; shift ;;
        --skip-diag)        SKIP_DIAG=1; shift ;;
        --skip-downstream)  SKIP_DOWNSTREAM=1; shift ;;
        --skip-pack)        SKIP_PACK=1; shift ;;
        --only-diag)        SKIP_TRAIN=1; shift ;;
        --regen-configs)    REGEN_CONFIGS=1; shift ;;
        --allow-partial)    ALLOW_PARTIAL=1; shift ;;
        --buckets)          BUCKETS_STR="$2"; shift 2 ;;
        --cuda)             CUDA_DEVICES="$2"; export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"; shift 2 ;;
        --num-processes)    NUM_PROCESSES="$2"; shift 2 ;;
        -h|--help)          sed -n '1,45p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# shellcheck disable=SC2206
BUCKETS=(${BUCKETS_STR})

log() { echo "$@"; }

run_cmd() {
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[R36 DRY-RUN] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} $*"
    else
        "$@"
    fi
}

selection_for_bucket() {
    case "$1" in
        train) echo "${SELECTION_TRAIN}" ;;
        val) echo "${SELECTION_VAL}" ;;
        *) echo "[R36] unknown bucket: $1" >&2; return 1 ;;
    esac
}

log "[R36] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  NUM_PROCESSES=${NUM_PROCESSES}"
log "[R36] BUCKETS=${BUCKETS[*]}  ALLOW_PARTIAL=${ALLOW_PARTIAL}"

GEN_ARGS=()
[[ -n "${DATASETS_ROOT:-}" ]] && GEN_ARGS+=(--data-root "${DATASETS_ROOT}")
[[ ${REGEN_CONFIGS} -eq 1 ]] && GEN_ARGS+=(--regen-bases)
[[ ${DRY_RUN} -eq 1 ]] && GEN_ARGS+=(--dry-run)
run_cmd "${PY}" -u scripts/stage_a_generator/round36_make_dynamics_configs.py "${GEN_ARGS[@]}"

preflight_fail=0
if [[ ${DRY_RUN} -eq 0 ]]; then
    for p in "${STAGE1_CFG}" "${STAGE1P5_CFG}"; do
        [[ ! -e "${p}" ]] && { echo "[R36 PREFLIGHT FAIL] missing config: ${p}"; preflight_fail=1; }
    done
    if [[ ${SKIP_TRAIN} -eq 1 || ${SKIP_DIAG} -eq 0 ]]; then
        for b in "${BUCKETS[@]}"; do
            sel="$(selection_for_bucket "${b}")" || preflight_fail=1
            [[ -n "${sel:-}" && ! -e "${sel}" ]] && { echo "[R36 PREFLIGHT FAIL] missing selection: ${sel}"; preflight_fail=1; }
        done
    fi
    if [[ ${SKIP_TRAIN} -eq 1 && ${SKIP_DIAG} -eq 0 ]]; then
        for p in "${STAGE1_CKPT}" "${STAGE1P5_CKPT}" "${PB1_CKPT}"; do
            [[ ! -e "${p}" ]] && { echo "[R36 PREFLIGHT FAIL] missing ckpt: ${p}"; preflight_fail=1; }
        done
    fi
fi
if [[ ${preflight_fail} -ne 0 ]]; then
    echo "[R36] FATAL preflight failures."
    exit 1
fi

train_one() {
    local stage="$1"; local vid="$2"; local cfg="$3"; local script="$4"; local final="$5"
    local log_file="${LOG_DIR}/train_${vid}.log"
    log
    log "================================================================"
    log "[$(date '+%F %T')] TRAIN ${vid}"
    log "    config: ${cfg}"
    log "    log:    ${log_file}"
    log "================================================================"

    if [[ ${SKIP_TRAIN} -eq 1 ]]; then
        log "[R36] --skip-train: ${vid}"
        return 0
    fi

    if [[ "${NUM_PROCESSES}" == "1" ]]; then
        cmd=("${PY}" -u "${script}" --config "${cfg}")
    else
        cmd=(accelerate launch
            --num_processes "${NUM_PROCESSES}" --multi_gpu --mixed_precision bf16
            "${script}" --config "${cfg}")
    fi

    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[R36 DRY-RUN] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${cmd[*]}"
        return 0
    fi

    set +e
    "${cmd[@]}" 2>&1 | tee "${log_file}"
    rc=${PIPESTATUS[0]}
    set -e
    if [[ ${rc} -ne 0 || ! -f "${final}" ]]; then
        log "[R36] ${stage} train failed for ${vid} (rc=${rc}, final=$([[ -f ${final} ]] && echo present || echo missing))"
        [[ "${ALLOW_PARTIAL}" == "1" ]] && return 0
        exit 1
    fi
}

train_one "stage1" "${STAGE1_VID}" "${STAGE1_CFG}" src/piano/training/train_stage1.py "${STAGE1_CKPT}"
train_one "stage1p5" "${STAGE1P5_VID}" "${STAGE1P5_CFG}" src/piano/training/train_stage1p5.py "${STAGE1P5_CKPT}"

run_temporal_diag() {
    local stage="$1"; local cfg="$2"; local pred_root="$3"; local tag="$4"; local bucket="$5"
    local sel; sel="$(selection_for_bucket "${bucket}")"
    local out_dir="${TEMPORAL_ROOT}/${tag}/${bucket}"
    local log_file="${LOG_DIR}/temporal_${tag}_${bucket}.log"
    mkdir -p "${out_dir}"
    local cmd=("${PY}" -u scripts/stage_a_generator/round36_temporal_dynamics_diag.py
        --stage "${stage}"
        --config "${cfg}"
        --generated-dir "${pred_root}"
        --selection-json "${sel}"
        --bucket "${bucket}"
        --out-md "${out_dir}/temporal_dynamics_summary.md"
        --out-json "${out_dir}/temporal_dynamics_stats.json")
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[R36 DRY-RUN] ${cmd[*]}"
    else
        "${cmd[@]}" 2>&1 | tee "${log_file}"
    fi
}

if [[ ${SKIP_DIAG} -eq 0 ]]; then
    DS_FLAGS=()
    [[ ${SKIP_DOWNSTREAM} -eq 1 ]] && DS_FLAGS+=(--skip-diag)

    log
    log "================================================================"
    log "[$(date '+%F %T')] DIAG Stage-1"
    log "================================================================"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[R36 DRY-RUN] ROUND31_DS_BUCKETS='${BUCKETS_STR}' ROUND31_DS_OUT_TAG=${STAGE1_TAG} bash scripts/stage_a_generator/run_round31_stage1_downstream_diag.sh ${DS_FLAGS[*]}"
    else
        ROUND31_DS_STAGE1_CFG="${STAGE1_CFG}" \
        ROUND31_DS_STAGE1_CKPT="${STAGE1_CKPT}" \
        ROUND31_DS_PB1_CKPT="${PB1_CKPT}" \
        ROUND31_DS_BUCKETS="${BUCKETS_STR}" \
        ROUND31_DS_OUT_TAG="${STAGE1_TAG}" \
        ROUND31_DS_ALLOW_PARTIAL="${ALLOW_PARTIAL}" \
            bash scripts/stage_a_generator/run_round31_stage1_downstream_diag.sh "${DS_FLAGS[@]}" \
            2>&1 | tee "${LOG_DIR}/downstream_stage1.log"
    fi
    for b in "${BUCKETS[@]}"; do
        run_temporal_diag "stage1" "${STAGE1_CFG}" "${STAGE1_SUB_ROOT}" "${STAGE1_VID}" "${b}"
    done

    log
    log "================================================================"
    log "[$(date '+%F %T')] DIAG Stage-1.5 oracle Stage-1 cond"
    log "================================================================"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[R36 DRY-RUN] ROUND32_DS_BUCKETS='${BUCKETS_STR}' ROUND32_DS_OUT_TAG=${STAGE1P5_ORACLE_TAG} bash scripts/stage_a_generator/run_round32_stage1p5_downstream_diag.sh ${DS_FLAGS[*]}"
    else
        ROUND32_DS_STAGE1P5_CFG="${STAGE1P5_CFG}" \
        ROUND32_DS_STAGE1P5_CKPT="${STAGE1P5_CKPT}" \
        ROUND32_DS_PB1_CKPT="${PB1_CKPT}" \
        ROUND32_DS_BUCKETS="${BUCKETS_STR}" \
        ROUND32_DS_OUT_TAG="${STAGE1P5_ORACLE_TAG}" \
        ROUND32_DS_ALLOW_PARTIAL="${ALLOW_PARTIAL}" \
            bash scripts/stage_a_generator/run_round32_stage1p5_downstream_diag.sh "${DS_FLAGS[@]}" \
            2>&1 | tee "${LOG_DIR}/downstream_stage1p5_oracle.log"
    fi
    for b in "${BUCKETS[@]}"; do
        run_temporal_diag "stage1p5" "${STAGE1P5_CFG}" "${STAGE1P5_ORACLE_SUB_ROOT}" "${STAGE1P5_VID}_oracle" "${b}"
    done

    log
    log "================================================================"
    log "[$(date '+%F %T')] DIAG Stage-1.5 generated Stage-1 cond"
    log "================================================================"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[R36 DRY-RUN] ROUND32_DS_BUCKETS='${BUCKETS_STR}' ROUND32_DS_UPSTREAM_DIR=${STAGE1_SUB_ROOT} ROUND32_DS_OUT_TAG=${STAGE1P5_GEN_TAG} bash scripts/stage_a_generator/run_round32_stage1p5_downstream_diag.sh ${DS_FLAGS[*]}"
    else
        ROUND32_DS_STAGE1P5_CFG="${STAGE1P5_CFG}" \
        ROUND32_DS_STAGE1P5_CKPT="${STAGE1P5_CKPT}" \
        ROUND32_DS_PB1_CKPT="${PB1_CKPT}" \
        ROUND32_DS_BUCKETS="${BUCKETS_STR}" \
        ROUND32_DS_OUT_TAG="${STAGE1P5_GEN_TAG}" \
        ROUND32_DS_UPSTREAM_DIR="${STAGE1_SUB_ROOT}" \
        ROUND32_DS_ALLOW_PARTIAL="${ALLOW_PARTIAL}" \
            bash scripts/stage_a_generator/run_round32_stage1p5_downstream_diag.sh "${DS_FLAGS[@]}" \
            2>&1 | tee "${LOG_DIR}/downstream_stage1p5_genstage1.log"
    fi
    for b in "${BUCKETS[@]}"; do
        run_temporal_diag "stage1p5" "${STAGE1P5_CFG}" "${STAGE1P5_GEN_SUB_ROOT}" "${STAGE1P5_VID}_genstage1" "${b}"
    done
else
    log "[R36] --skip-diag: diagnostics skipped"
fi

if [[ ${SKIP_PACK} -eq 0 ]]; then
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[R36 DRY-RUN] ROUND36_BUCKETS='${BUCKETS_STR}' bash scripts/stage_a_generator/pack_round36_dynamics_sync.sh --allow-missing"
    else
        ROUND36_BUCKETS="${BUCKETS_STR}" \
        ROUND36_ALLOW_MISSING="${ALLOW_PARTIAL}" \
            bash scripts/stage_a_generator/pack_round36_dynamics_sync.sh \
            2>&1 | tee "${LOG_DIR}/pack_sync.log"
    fi
fi

log
log "================================================================"
log "Round-36 dynamics training + diagnostics complete."
log "================================================================"
