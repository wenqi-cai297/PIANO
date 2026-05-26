#!/usr/bin/env bash
# Round-29 full-data A2 vs A3 launcher.
#
# Per Codex review (analyses/2026-05-27_codex_round29_review.md P0.2 +
# Tier-2 ordering): the A-group 48-clip result did not crown one
# injection — A2 won body, A3 won gait, A0 won contact. The decisive
# experiment is full-data A2 vs A3 with heldout validation. The selected
# mainline then feeds Tier-2 and the B/C/D/E content matrix.
#
# Phase 1 (TRAIN): 2 variants sequential, each uses all GPUs via
#   accelerate. Schedule = 80 ep, val_every=5, save_every=10 (matches
#   v27 FULL_DATA). heldout val (val_on_train_subset=false).
# Phase 2 (DIAG): 2 variants × 3 diag = 6 tasks, parallel across N GPU
#   workers. Diag selection = the 48-clip balanced subset (same as
#   A-group) so contact/gait/body numbers are directly comparable.
#
# Usage:
#   bash scripts/stage_b_generator/run_a2_a3_full_data.sh
#   bash scripts/stage_b_generator/run_a2_a3_full_data.sh --dry-run
#   bash scripts/stage_b_generator/run_a2_a3_full_data.sh --only a2
#   bash scripts/stage_b_generator/run_a2_a3_full_data.sh --only a3
#   bash scripts/stage_b_generator/run_a2_a3_full_data.sh --skip-train
#
# Environment overrides:
#   ROUND29_NUM_PROCESSES=N               accelerate --num_processes (default: nvidia-smi -L)
#   ROUND29_PARALLEL_DIAG_WORKERS=N       diag workers (default: NUM_PROCESSES)
#   ROUND29_DIAG_CKPT_NAME=best_val.pt    diag ckpt filename (default: final.pt)
#   ROUND29_SINGLE_GPU=1                  force single-GPU train
#   WANDB_DISABLED=1                      disable wandb (jsonl-only metrics)

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
SKIP_TRAIN=0
SKIP_EVAL=0
SINGLE_GPU="${ROUND29_SINGLE_GPU:-0}"
DIAG_CKPT_NAME="${ROUND29_DIAG_CKPT_NAME:-final.pt}"
# Two 48-clip selections — diag runs on both for fair comparison:
#   train  = in-distribution sanity (same selection a-group used)
#   val    = heldout-val generalization (built by round29_build_val_diag_subset.py)
SELECTION_TRAIN="analyses/round27_tier0_train_indices_48_balanced.json"
SELECTION_VAL="analyses/round29_val_diag_indices_48_balanced.json"

if [[ -n "${ROUND29_NUM_PROCESSES:-}" ]]; then
    NUM_PROCESSES="${ROUND29_NUM_PROCESSES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_PROCESSES="$(nvidia-smi -L | wc -l)"
    [[ "${NUM_PROCESSES}" -lt 1 ]] && NUM_PROCESSES=1
else
    NUM_PROCESSES=2
fi
PARALLEL_DIAG_WORKERS="${ROUND29_PARALLEL_DIAG_WORKERS:-${NUM_PROCESSES}}"

LOG_DIR="runs/round29_a2_a3_full_data"
mkdir -p "${LOG_DIR}"
PY="${PY:-python}"

# Variant table: ID, config, output dir.
declare -a VARIANT_IDS=("a2" "a3")
declare -A VARIANT_CFG=(
    [a2]="configs/training/anchordiff_a2_full_data.yaml"
    [a3]="configs/training/anchordiff_a3_full_data.yaml"
)
declare -A VARIANT_OUT=(
    [a2]="runs/training/stageB_anchordiff_a2_full_data"
    [a3]="runs/training/stageB_anchordiff_a3_full_data"
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)          ONLY="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --skip-train)    SKIP_TRAIN=1; shift ;;
        --skip-eval)     SKIP_EVAL=1; shift ;;
        --diag-ckpt-name) DIAG_CKPT_NAME="$2"; shift 2 ;;
        --num-processes) NUM_PROCESSES="$2"; shift 2 ;;
        --parallel-diag-workers) PARALLEL_DIAG_WORKERS="$2"; shift 2 ;;
        --single-gpu)    SINGLE_GPU=1; shift ;;
        -h|--help)
            sed -n '1,30p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

echo "[A23] NUM_PROCESSES=${NUM_PROCESSES}  PARALLEL_DIAG_WORKERS=${PARALLEL_DIAG_WORKERS}  DIAG_CKPT_NAME=${DIAG_CKPT_NAME}"

# Build SELECTED list.
SELECTED=()
if [[ -n "${ONLY}" ]]; then
    IFS=',' read -ra wants <<< "${ONLY}"
    for w in "${wants[@]}"; do
        if [[ -z "${VARIANT_CFG[$w]:-}" ]]; then
            echo "Unknown variant: $w (valid: a2, a3)" >&2; exit 2
        fi
        SELECTED+=("$w")
    done
else
    SELECTED=("${VARIANT_IDS[@]}")
fi

echo "[A23] Variants: ${SELECTED[*]}"

# Preflight.
if [[ ${DRY_RUN} -eq 0 ]]; then
    preflight_fail=0
    for VID in "${SELECTED[@]}"; do
        CFG="${VARIANT_CFG[$VID]}"
        if [[ ! -e "${CFG}" ]]; then
            echo "    [${VID}] missing config: ${CFG}"
            preflight_fail=1
        fi
    done
    for sel in "${SELECTION_TRAIN}" "${SELECTION_VAL}"; do
        if [[ ! -e "${sel}" ]]; then
            echo "    missing selection JSON: ${sel}"
            if [[ "${sel}" == "${SELECTION_VAL}" ]]; then
                echo "    -> generate it with: python scripts/stage_b_generator/round29_build_val_diag_subset.py --config ${VARIANT_CFG[${SELECTED[0]}]}"
            fi
            preflight_fail=1
        fi
    done
    # Dataset roots — parse first config.
    if [[ ${SKIP_TRAIN} -eq 0 ]]; then
        FIRST_CFG="${VARIANT_CFG[${SELECTED[0]}]}"
        BAD="$("${PY}" -c "
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(open(sys.argv[1]))
for ds in (cfg.get('data', {}).get('datasets') or []):
    root = ds.get('root', '')
    if root and not Path(root).exists():
        print(f\"{ds.get('name')}={root}\")
" "${FIRST_CFG}")"
        if [[ -n "${BAD}" ]]; then
            while IFS= read -r br; do
                echo "    dataset root not on disk: ${br}"
            done <<< "${BAD}"
            echo "    -> edit the dataset paths inside the YAMLs (these are tracked, hand-curated configs)"
            preflight_fail=1
        fi
    fi
    if [[ ${preflight_fail} -ne 0 ]]; then
        echo "[A23] FATAL preflight failures."
        exit 1
    fi
fi

# PHASE 1: TRAIN sequentially (each train uses all GPUs).
TRAINED_OK=()
for VID in "${SELECTED[@]}"; do
    CFG="${VARIANT_CFG[$VID]}"
    OUTDIR="${VARIANT_OUT[$VID]}"
    LOG="${LOG_DIR}/${VID}.log"
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] TRAIN ${VID}"
    echo "    config: ${CFG}"
    echo "    output: ${OUTDIR}"
    echo "    log:    ${LOG}"
    echo "================================================================"

    if [[ ${SKIP_TRAIN} -eq 0 ]]; then
        if [[ "${SINGLE_GPU}" == "1" || "${NUM_PROCESSES}" == "1" ]]; then
            TRAIN_CMD=("${PY}" -u src/piano/training/train_anchordiff.py --config "${CFG}")
        else
            TRAIN_CMD=(accelerate launch
                --num_processes "${NUM_PROCESSES}" --multi_gpu --mixed_precision bf16
                src/piano/training/train_anchordiff.py --config "${CFG}")
        fi
        if [[ ${DRY_RUN} -eq 1 ]]; then
            echo "[A23 DRY-RUN ${VID} TRAIN]"
            echo "    \$ ${TRAIN_CMD[*]}"
            TRAINED_OK+=("${VID}")
        else
            if "${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG}"; then
                TRAINED_OK+=("${VID}")
            else
                echo "[A23] WARN: training failed for ${VID}; skipping diag"
            fi
        fi
    else
        echo "--skip-train: skipping training for ${VID}"
        TRAINED_OK+=("${VID}")
    fi
done

# PHASE 2: DIAG parallel across GPUs.
if [[ ${SKIP_EVAL} -eq 1 ]]; then
    echo
    echo "--skip-eval: skipping diag"
elif [[ ${#TRAINED_OK[@]} -eq 0 ]]; then
    echo "[A23] No variants succeeded training; no diag to run."
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] DIAG PHASE (workers=${PARALLEL_DIAG_WORKERS})"
    echo "================================================================"

    TASK_QUEUE="$(mktemp -t a23_diag_tasks.XXXXXX)"
    trap "rm -f '${TASK_QUEUE}' '${TASK_QUEUE}.lock'" EXIT

    for VID in "${TRAINED_OK[@]}"; do
        CFG="${VARIANT_CFG[$VID]}"
        OUTDIR="${VARIANT_OUT[$VID]}"
        CKPT_PATH="${OUTDIR}/${DIAG_CKPT_NAME}"
        if [[ ! -e "${CKPT_PATH}" && ${DRY_RUN} -eq 0 ]]; then
            echo "[A23] WARN: diag ckpt missing: ${CKPT_PATH} (skipped)"
            continue
        fi
        # 3 diag kinds × 2 selection buckets = 6 tasks per variant.
        for kind in sustained_contact gait body_action; do
            case "${kind}" in
                sustained_contact) DIAG_SCRIPT="scripts/stage_b_generator/round26_sustained_contact_diag.py" ;;
                gait)              DIAG_SCRIPT="scripts/stage_b_generator/round26_gait_diag.py" ;;
                body_action)       DIAG_SCRIPT="scripts/stage_b_generator/round28_body_action_diag.py" ;;
            esac
            for sublabel in train val; do
                case "${sublabel}" in
                    train) SUBSET_PATH="${SELECTION_TRAIN}" ;;
                    val)   SUBSET_PATH="${SELECTION_VAL}" ;;
                esac
                OUT_DIR="analyses/round29_full_${VID}_diag_${kind}_${sublabel}"
                mkdir -p "${OUT_DIR}"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "${VID}" "${kind}_${sublabel}" "${DIAG_SCRIPT}" "${CFG}" "${CKPT_PATH}" "${SUBSET_PATH}" "${sublabel}" \
                    >> "${TASK_QUEUE}"
            done
        done
    done

    N_TASKS="$(wc -l < "${TASK_QUEUE}")"
    echo "[A23] ${N_TASKS} diag tasks queued; launching ${PARALLEL_DIAG_WORKERS} GPU workers..."

    if [[ ${DRY_RUN} -eq 1 ]]; then
        IDX=0
        while IFS=$'\t' read -r VID KIND DIAG_SCRIPT CFG CKPT_PATH SUBSET BUCKET; do
            GPU=$((IDX % PARALLEL_DIAG_WORKERS))
            OUT_DIR="analyses/round29_full_${VID}_diag_${KIND}"
            echo "[A23 DRY-RUN [GPU ${GPU}] ${VID} DIAG/${KIND}]"
            echo "    \$ CUDA_VISIBLE_DEVICES=${GPU} ${PY} -u ${DIAG_SCRIPT} --config ${CFG} --ckpt ${CKPT_PATH} --selection-json ${SUBSET} --output-dir ${OUT_DIR} --bucket ${BUCKET}"
            IDX=$((IDX + 1))
        done < "${TASK_QUEUE}"
    else
        QUEUE_LOCK="${TASK_QUEUE}.lock"
        FAIL_LOG="${TASK_QUEUE}.fail"
        : > "${QUEUE_LOCK}"
        : > "${FAIL_LOG}"
        trap "rm -f '${TASK_QUEUE}' '${QUEUE_LOCK}' '${FAIL_LOG}'" EXIT
        WORKER_PIDS=()
        for ((W = 0; W < PARALLEL_DIAG_WORKERS; W++)); do
            (
                while true; do
                    TASK_LINE="$(
                        flock -x "${QUEUE_LOCK}" -c "
                            line=\$(head -n 1 '${TASK_QUEUE}')
                            if [[ -n \"\$line\" ]]; then
                                sed -i '1d' '${TASK_QUEUE}'
                                echo \"\$line\"
                            fi
                        "
                    )"
                    [[ -z "${TASK_LINE}" ]] && break
                    IFS=$'\t' read -r VID KIND DIAG_SCRIPT CFG CKPT_PATH SUBSET BUCKET <<< "${TASK_LINE}"
                    OUT_DIR="analyses/round29_full_${VID}_diag_${KIND}"
                    DIAG_LOG="${LOG_DIR}/${VID}_diag_${KIND}.log"
                    T0=$(date +%s)
                    echo "[A23] [GPU ${W}] START ${VID}/${KIND}  log: ${DIAG_LOG}"
                    : > "${DIAG_LOG}"
                    set +e
                    CUDA_VISIBLE_DEVICES="${W}" \
                        "${PY}" -u "${DIAG_SCRIPT}" \
                        --config "${CFG}" \
                        --ckpt "${CKPT_PATH}" \
                        --selection-json "${SUBSET}" \
                        --output-dir "${OUT_DIR}" \
                        --bucket "${BUCKET}" \
                        > "${DIAG_LOG}" 2>&1
                    RC=$?
                    set -e
                    T1=$(date +%s)
                    if [[ ${RC} -eq 0 ]]; then
                        echo "[A23] [GPU ${W}] DONE  ${VID}/${KIND}  ($((T1 - T0))s)"
                    else
                        flock -x "${QUEUE_LOCK}" -c "echo '${VID}/${KIND} rc=${RC}' >> '${FAIL_LOG}'"
                        echo "[A23] [GPU ${W}] FAIL  ${VID}/${KIND}  rc=${RC} ($((T1 - T0))s)  log: ${DIAG_LOG}"
                        echo "[A23] [GPU ${W}] tail of ${DIAG_LOG}:"
                        tail -n 20 "${DIAG_LOG}" | sed "s/^/[A23] [GPU ${W}]   /"
                    fi
                done
            ) &
            WORKER_PIDS+=($!)
        done
        for pid in "${WORKER_PIDS[@]}"; do
            wait "${pid}" || true
        done
        rm -f "${QUEUE_LOCK}"
        N_FAIL=$(wc -l < "${FAIL_LOG}" 2>/dev/null || echo 0)
        if [[ ${N_FAIL} -gt 0 ]]; then
            echo "[A23] ${N_FAIL}/${N_TASKS} diag tasks FAILED:"
            sed 's/^/[A23]   /' "${FAIL_LOG}"
        else
            echo "[A23] all ${N_TASKS} diag tasks succeeded."
        fi
    fi
fi

# Pack results.
if [[ ${DRY_RUN} -eq 0 ]]; then
    STAMP="$(date +%Y%m%d_%H%M%S)"
    PACK="analyses/round29_a2_a3_full_data_results_${STAMP}.tar.gz"
    tar -czf "${PACK}" \
        analyses/round29_full_a2_diag_* \
        analyses/round29_full_a3_diag_* 2>/dev/null || true
    echo "[A23] Packed ${PACK}"
fi
