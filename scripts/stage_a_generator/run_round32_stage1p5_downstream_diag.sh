#!/usr/bin/env bash
# Round-32 Stage-1.5 downstream-coupling diagnostic launcher (C in design doc).
#
# Pipes Stage-1.5's sampled (C41, S4) into frozen PB1, with the oracle
# Stage-1 stage1_coarse left intact. Comparing to PB1's oracle-cond
# metrics tells us how much Stage-1.5's imperfect output degrades
# downstream Stage-2 quality, independently of Stage-1.

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
SKIP_SAMPLE=0
SKIP_DIAG=0
ALLOW_PARTIAL="${ROUND32_DS_ALLOW_PARTIAL:-0}"
SEED="${ROUND32_DS_SEED:-42}"
CFG_SCALE="${ROUND32_DS_CFG_SCALE:-1.0}"
SAMPLER="${ROUND32_DS_SAMPLER:-ddim_eta0}"
STAGE1P5_CFG="${ROUND32_DS_STAGE1P5_CFG:-configs/training/stage1p5_interaction_v0.yaml}"
STAGE1P5_CKPT="${ROUND32_DS_STAGE1P5_CKPT:-runs/training/stage1p5_interaction_v0/final.pt}"
# Optional: when set, Stage-1.5 sampling pulls stage1_coarse from this dir
# (Stage-1's generated cache) instead of the oracle motion-derived path.
# This is the "generated Stage-1 cond" eval mode for R34 (per ChatGPT
# followup §7.2). Leave empty for default "oracle Stage-1 cond" mode.
UPSTREAM_DIR="${ROUND32_DS_UPSTREAM_DIR:-}"
PB1_VARIANT="r29_pb_a1_adaln_s4"
PB1_CFG="configs/training/anchordiff_${PB1_VARIANT}.yaml"
PB1_CKPT="${ROUND32_DS_PB1_CKPT:-runs/training/stageB_anchordiff_${PB1_VARIANT}/final.pt}"
BUCKETS_STR="${ROUND32_DS_BUCKETS:-train val}"
SELECTION_TRAIN="analyses/round27_tier0_train_indices_48_balanced.json"
SELECTION_VAL="analyses/round29_val_diag_indices_48_balanced.json"
OUT_TAG="${ROUND32_DS_OUT_TAG:-}"
SUB_DIR_ROOT="analyses/round32_stage1p5_substitute_conds${OUT_TAG}"
DIAG_DIR_ROOT="analyses/round32_stage1p5_downstream_diag${OUT_TAG}"
LOG_DIR="runs/round32_stage1p5_downstream${OUT_TAG}"
mkdir -p "${LOG_DIR}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)           ONLY="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=1; shift ;;
        --skip-sample)    SKIP_SAMPLE=1; shift ;;
        --skip-diag)      SKIP_DIAG=1; shift ;;
        -h|--help)        sed -n '1,40p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python";
    elif command -v python3 >/dev/null 2>&1; then PY="python3";
    else echo "[DS32] FATAL: neither python nor python3 was found" >&2; exit 127; fi
fi
if [[ -n "${ONLY}" ]]; then BUCKETS_STR="${ONLY}"; fi
# shellcheck disable=SC2206
BUCKETS=(${BUCKETS_STR})

echo "[DS32] STAGE1P5_CKPT=${STAGE1P5_CKPT}"
echo "[DS32] PB1_CKPT=${PB1_CKPT}"
echo "[DS32] BUCKETS=${BUCKETS[*]}  SAMPLER=${SAMPLER}  SEED=${SEED}  CFG_SCALE=${CFG_SCALE}"
if [[ -n "${UPSTREAM_DIR}" ]]; then
    echo "[DS32] UPSTREAM_DIR=${UPSTREAM_DIR}  (generated Stage-1 cond mode)"
else
    echo "[DS32] UPSTREAM_DIR=(empty)  (oracle Stage-1 cond mode)"
fi

# Preflight. Under --dry-run we tolerate missing ckpts/selections so a
# laptop can sanity-check env passthrough.
preflight_fail=0
for p in "${STAGE1P5_CFG}" "${PB1_CFG}"; do
    [[ ! -e "${p}" ]] && { echo "[DS32 PREFLIGHT FAIL] missing config: ${p}"; preflight_fail=1; }
done
if [[ ${DRY_RUN} -eq 0 ]]; then
    for p in "${STAGE1P5_CKPT}" "${PB1_CKPT}"; do
        [[ ! -e "${p}" ]] && { echo "[DS32 PREFLIGHT FAIL] missing ckpt: ${p}"; preflight_fail=1; }
    done
    for b in "${BUCKETS[@]}"; do
        case "${b}" in
            train) sel="${SELECTION_TRAIN}" ;;
            val)   sel="${SELECTION_VAL}"   ;;
            *) echo "[DS32 PREFLIGHT FAIL] unknown bucket: ${b}"; preflight_fail=1; continue ;;
        esac
        [[ ! -e "${sel}" ]] && { echo "[DS32 PREFLIGHT FAIL] missing selection: ${sel}"; preflight_fail=1; }
    done
fi
for s in scripts/stage_b_generator/round26_sustained_contact_diag.py \
         scripts/stage_b_generator/round26_gait_diag.py \
         scripts/stage_b_generator/round28_body_action_diag.py \
         scripts/stage_b_generator/round29_g1_soft_stance_diag.py \
         scripts/stage_a_generator/sample_substitute_conds_cli.py; do
    [[ ! -e "${s}" ]] && { echo "[DS32 PREFLIGHT FAIL] missing script: ${s}"; preflight_fail=1; }
done
if [[ ${preflight_fail} -ne 0 ]]; then echo "[DS32] FATAL preflight failures."; exit 1; fi

# Phase 1: Sample Stage-1.5 (uses oracle stage1_coarse cond, NO upstream-dir).
for b in "${BUCKETS[@]}"; do
    case "${b}" in train) sel="${SELECTION_TRAIN}" ;; val) sel="${SELECTION_VAL}" ;; esac
    SUB_DIR="${SUB_DIR_ROOT}/${b}"
    LOG="${LOG_DIR}/sample_stage1p5_${b}.log"
    if [[ ${SKIP_SAMPLE} -eq 1 ]]; then
        echo "[DS32] --skip-sample: skipping sampling for ${b}"; continue
    fi
    echo
    echo "================================================================"
    if [[ -n "${UPSTREAM_DIR}" ]]; then
        MODE_LABEL="generated Stage-1 cond from ${UPSTREAM_DIR}/${b}"
    else
        MODE_LABEL="oracle Stage-1 cond"
    fi
    echo "[$(date '+%F %T')] PHASE 1: SAMPLE Stage-1.5 (${MODE_LABEL}) → ${SUB_DIR}"
    echo "================================================================"
    SAMPLE_CMD=("${PY}" -u scripts/stage_a_generator/sample_substitute_conds_cli.py
        --stage stage1p5
        --config "${STAGE1P5_CFG}"
        --ckpt "${STAGE1P5_CKPT}"
        --selection-json "${sel}"
        --bucket "${b}"
        --out-dir "${SUB_DIR}"
        --seed "${SEED}"
        --cfg-scale "${CFG_SCALE}"
        --sampler "${SAMPLER}")
    if [[ -n "${UPSTREAM_DIR}" ]]; then
        SAMPLE_CMD+=(--upstream-dir "${UPSTREAM_DIR}/${b}")
    fi
    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "[DS32 DRY-RUN]"; echo "    \$ ${SAMPLE_CMD[*]}"
    else
        "${SAMPLE_CMD[@]}" 2>&1 | tee "${LOG}"
    fi
done

# Phase 2: Run 4 diags with substitute_conds_dir.
run_diag() {
    local KIND="$1"; local BUCKET="$2"; local SUB_DIR="$3"
    case "${BUCKET}" in train) sel="${SELECTION_TRAIN}" ;; val) sel="${SELECTION_VAL}" ;; esac
    case "${KIND}" in
        sustained_contact) SCRIPT="scripts/stage_b_generator/round26_sustained_contact_diag.py" ;;
        gait)              SCRIPT="scripts/stage_b_generator/round26_gait_diag.py" ;;
        body_action)       SCRIPT="scripts/stage_b_generator/round28_body_action_diag.py" ;;
        g1_soft_stance)    SCRIPT="scripts/stage_b_generator/round29_g1_soft_stance_diag.py" ;;
        *) echo "[DS32 DIAG] unknown kind ${KIND}"; return 1 ;;
    esac
    OUT_DIR="${DIAG_DIR_ROOT}/${KIND}_${BUCKET}"
    LOG="${LOG_DIR}/diag_${KIND}_${BUCKET}.log"
    mkdir -p "${OUT_DIR}"
    CMD=("${PY}" -u "${SCRIPT}"
        --config "${PB1_CFG}" --ckpt "${PB1_CKPT}"
        --selection-json "${sel}"
        --output-dir "${OUT_DIR}"
        --bucket "${BUCKET}"
        --substitute-conds-dir "${SUB_DIR}"
        --cfg-scale "${CFG_SCALE}" --seed "${SEED}")
    echo "[DS32 DIAG START] ${KIND} ${BUCKET}"
    if [[ ${DRY_RUN} -eq 1 ]]; then echo "    \$ ${CMD[*]}"; return 0; fi
    if ! "${CMD[@]}" 2>&1 | tee "${LOG}"; then
        if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
            echo "[DS32] WARN: ${KIND} ${BUCKET} failed; continuing (ALLOW_PARTIAL=1)"
        else
            echo "[DS32] FATAL: ${KIND} ${BUCKET} failed."; return 1
        fi
    fi
}

if [[ ${SKIP_DIAG} -eq 1 ]]; then
    echo "[DS32] --skip-diag: skipping Phase 2"
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE 2: DIAG (4 kinds × ${#BUCKETS[@]} buckets)"
    echo "================================================================"
    for b in "${BUCKETS[@]}"; do
        SUB_DIR="${SUB_DIR_ROOT}/${b}"
        for KIND in sustained_contact gait body_action g1_soft_stance; do
            run_diag "${KIND}" "${b}" "${SUB_DIR}"
        done
    done
fi

# Phase 3: Pack.
if [[ ${DRY_RUN} -eq 0 ]]; then
    STAMP=$(date +%Y%m%d_%H%M%S)
    TARBALL="analyses/round32_stage1p5_downstream_results${OUT_TAG}_${STAMP}.tar.gz"
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PACK -> ${TARBALL}"
    echo "================================================================"
    PACK_TARGETS=()
    for b in "${BUCKETS[@]}"; do
        for KIND in sustained_contact gait body_action g1_soft_stance; do
            D="${DIAG_DIR_ROOT}/${KIND}_${b}"
            [[ -d "${D}" ]] && PACK_TARGETS+=("${D}")
        done
    done
    [[ -d "${SUB_DIR_ROOT}" ]] && PACK_TARGETS+=("${SUB_DIR_ROOT}")
    [[ -d "${LOG_DIR}" ]] && PACK_TARGETS+=("${LOG_DIR}")
    if [[ ${#PACK_TARGETS[@]} -gt 0 ]]; then
        tar -czf "${TARBALL}" "${PACK_TARGETS[@]}"
        SIZE=$(du -h "${TARBALL}" | cut -f1)
        echo "wrote ${TARBALL}  (${SIZE})"
    else
        echo "[DS32 PACK] nothing to pack"
    fi
fi

echo
echo "================================================================"
echo "Round-32 Stage-1.5 downstream diag complete."
echo "================================================================"
