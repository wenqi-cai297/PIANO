#!/usr/bin/env bash
# Round-29 Tier-2 ablation launcher.
#
# Trains 3 single-variable-change variants from r29_a0_input_add baseline:
#   r29_t2_no_init_pose   — init_pose_dim=0
#   r29_t2_no_text        — text_dim=0
#   r29_t2_no_dense_pos   — pos_loss_weight=0
#
# Phase 1 (TRAIN): 3 variants sequential, each uses all GPUs via accelerate.
# Phase 2 (DIAG): 3 variants × 3 diag = 9 tasks, parallel across N GPU workers.
#
# Usage:
#   bash scripts/stage_b_generator/run_round29_tier2_ablations.sh
#   bash scripts/stage_b_generator/run_round29_tier2_ablations.sh --dry-run
#   bash scripts/stage_b_generator/run_round29_tier2_ablations.sh --only r29_t2_no_init_pose
#   bash scripts/stage_b_generator/run_round29_tier2_ablations.sh --skip-train
#
# Environment overrides:
#   DATASETS_ROOT=...                     dataset root (default = dev Windows path)
#   ROUND29_NUM_PROCESSES=N               accelerate --num_processes (default: nvidia-smi -L)
#   ROUND29_PARALLEL_DIAG_WORKERS=N       diag-phase workers (default: NUM_PROCESSES)
#   ROUND29_DIAG_CKPT_NAME=best_val.pt    diag ckpt filename (default: final.pt)
#   ROUND29_SINGLE_GPU=1                  force single-GPU train (no accelerate)

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
SKIP_TRAIN=0
SKIP_EVAL=0
SINGLE_GPU="${ROUND29_SINGLE_GPU:-0}"
DIAG_CKPT_NAME="${ROUND29_DIAG_CKPT_NAME:-final.pt}"

if [[ -n "${ROUND29_NUM_PROCESSES:-}" ]]; then
    NUM_PROCESSES="${ROUND29_NUM_PROCESSES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_PROCESSES="$(nvidia-smi -L | wc -l)"
    [[ "${NUM_PROCESSES}" -lt 1 ]] && NUM_PROCESSES=1
else
    NUM_PROCESSES=2
fi
PARALLEL_DIAG_WORKERS="${ROUND29_PARALLEL_DIAG_WORKERS:-${NUM_PROCESSES}}"

MANIFEST="analyses/round29_tier2_ablation_manifest.json"
LOG_DIR="runs/round29_tier2_ablation"
mkdir -p "${LOG_DIR}"
PY="${PY:-python}"

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
            sed -n '1,28p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

echo "[T2] NUM_PROCESSES=${NUM_PROCESSES}  PARALLEL_DIAG_WORKERS=${PARALLEL_DIAG_WORKERS}  DIAG_CKPT_NAME=${DIAG_CKPT_NAME}"

# (1) Generate manifest + configs if missing or stale.
if [[ ! -f "${MANIFEST}" ]]; then
    echo "[T2] Manifest missing — running config generator..."
    GEN_ARGS=()
    [[ -n "${DATASETS_ROOT:-}" ]] && GEN_ARGS+=(--data-root "${DATASETS_ROOT}")
    "${PY}" scripts/stage_b_generator/round29_make_tier2_ablation_configs.py "${GEN_ARGS[@]}"
fi

# (2) Pick variants from manifest.
PICK_SCRIPT='
import json, sys
m = json.load(open(sys.argv[1]))
only = sys.argv[2]
want_only = set(only.split(",")) if only else None
for v in m["variants"]:
    if want_only is not None and v["variant_id"] not in want_only: continue
    print(v["variant_id"], v["config_path"], v["output_dir"],
          v["diag_train_subset"], v["diag_val_subset"])
'
VARIANTS="$("${PY}" -c "${PICK_SCRIPT}" "${MANIFEST}" "${ONLY}")"

if [[ -z "${VARIANTS}" ]]; then
    echo "[T2] no variants matched only='${ONLY}'"
    exit 0
fi

echo "[T2] Variants to process:"
echo "${VARIANTS}"

# (3) Preflight.
if [[ ${DRY_RUN} -eq 0 ]]; then
    preflight_fail=0
    while IFS=' ' read -r VID CFG OUTDIR TRAIN_SUBSET VAL_SUBSET; do
        [[ -z "${VID}" ]] && continue
        for p in "${CFG}" "${TRAIN_SUBSET}" "${VAL_SUBSET}"; do
            if [[ ! -e "${p}" ]]; then
                echo "    [${VID}] missing: ${p}"
                preflight_fail=1
            fi
        done
        # Dataset roots — parse YAML.
        if [[ ${SKIP_TRAIN} -eq 0 && -e "${CFG}" ]]; then
            BAD="$("${PY}" -c "
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(open(sys.argv[1]))
for ds in (cfg.get('data', {}).get('datasets') or []):
    root = ds.get('root', '')
    if root and not Path(root).exists():
        print(f\"{ds.get('name')}={root}\")
" "${CFG}")"
            if [[ -n "${BAD}" ]]; then
                while IFS= read -r br; do
                    echo "    [${VID}] dataset root not on disk: ${br}"
                done <<< "${BAD}"
                echo "    [${VID}]   -> re-run generator with --data-root <correct path> or export DATASETS_ROOT=..."
                preflight_fail=1
            fi
        fi
    done <<< "${VARIANTS}"
    if [[ ${preflight_fail} -ne 0 ]]; then
        echo "[T2] FATAL preflight failures."
        exit 1
    fi
fi

# (4) PHASE 1: TRAIN sequentially.
TRAINED_OK=""
while IFS=' ' read -r VID CFG OUTDIR TRAIN_SUBSET VAL_SUBSET; do
    [[ -z "${VID}" ]] && continue
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
            echo "[T2 DRY-RUN ${VID} TRAIN]"
            echo "    \$ ${TRAIN_CMD[*]}"
            TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR} ${TRAIN_SUBSET} ${VAL_SUBSET}"$'\n'
        else
            if "${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG}"; then
                TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR} ${TRAIN_SUBSET} ${VAL_SUBSET}"$'\n'
            else
                echo "[T2] WARN: training failed for ${VID}; skipping diag"
            fi
        fi
    else
        echo "--skip-train: skipping training for ${VID}"
        TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR} ${TRAIN_SUBSET} ${VAL_SUBSET}"$'\n'
    fi
done <<< "${VARIANTS}"

# (5) PHASE 2: DIAG parallel across GPUs.
if [[ ${SKIP_EVAL} -eq 1 ]]; then
    echo
    echo "--skip-eval: skipping diag"
elif [[ -z "${TRAINED_OK}" ]]; then
    echo "[T2] No variants succeeded training; no diag to run."
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] DIAG PHASE (workers=${PARALLEL_DIAG_WORKERS})"
    echo "================================================================"

    TASK_QUEUE="$(mktemp -t t2_diag_tasks.XXXXXX)"
    trap "rm -f '${TASK_QUEUE}' '${TASK_QUEUE}.lock'" EXIT

    while IFS=' ' read -r VID CFG OUTDIR TRAIN_SUBSET VAL_SUBSET; do
        [[ -z "${VID}" ]] && continue
        CKPT_PATH="${OUTDIR}/${DIAG_CKPT_NAME}"
        if [[ ! -e "${CKPT_PATH}" && ${DRY_RUN} -eq 0 ]]; then
            echo "[T2] WARN: diag ckpt missing: ${CKPT_PATH} (skipped)"
            continue
        fi
        # 3 diag kinds × 2 selection subsets = 6 tasks per variant.
        # Subset labels: "train" = in-distribution sanity, "val" = generalization.
        for kind in sustained_contact gait body_action; do
            case "${kind}" in
                sustained_contact) DIAG_SCRIPT="scripts/stage_b_generator/round26_sustained_contact_diag.py" ;;
                gait)              DIAG_SCRIPT="scripts/stage_b_generator/round26_gait_diag.py" ;;
                body_action)       DIAG_SCRIPT="scripts/stage_b_generator/round28_body_action_diag.py" ;;
            esac
            for sublabel in train val; do
                case "${sublabel}" in
                    train) SUBSET_PATH="${TRAIN_SUBSET}" ;;
                    val)   SUBSET_PATH="${VAL_SUBSET}" ;;
                esac
                OUT_DIR="analyses/round29_${VID}_diag_${kind}_${sublabel}"
                mkdir -p "${OUT_DIR}"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "${VID}" "${kind}_${sublabel}" "${DIAG_SCRIPT}" "${CFG}" "${CKPT_PATH}" "${SUBSET_PATH}" "${sublabel}" \
                    >> "${TASK_QUEUE}"
            done
        done
    done <<< "${TRAINED_OK}"

    N_TASKS="$(wc -l < "${TASK_QUEUE}")"
    echo "[T2] ${N_TASKS} diag tasks queued; launching ${PARALLEL_DIAG_WORKERS} GPU workers..."

    if [[ ${DRY_RUN} -eq 1 ]]; then
        IDX=0
        while IFS=$'\t' read -r VID KIND DIAG_SCRIPT CFG CKPT_PATH SUBSET BUCKET; do
            GPU=$((IDX % PARALLEL_DIAG_WORKERS))
            OUT_DIR="analyses/round29_${VID}_diag_${KIND}"
            echo "[T2 DRY-RUN [GPU ${GPU}] ${VID} DIAG/${KIND}]"
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
                    OUT_DIR="analyses/round29_${VID}_diag_${KIND}"
                    DIAG_LOG="${LOG_DIR}/${VID}_diag_${KIND}.log"
                    T0=$(date +%s)
                    echo "[T2] [GPU ${W}] START ${VID}/${KIND}  log: ${DIAG_LOG}"
                    # Truncate per task so re-runs do not accumulate stale tracebacks.
                    : > "${DIAG_LOG}"
                    # set +e so set -e does not kill the worker subshell on
                    # non-zero diag exit; capture RC for DONE/FAIL reporting.
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
                        echo "[T2] [GPU ${W}] DONE  ${VID}/${KIND}  ($((T1 - T0))s)"
                    else
                        # Record the failure for the launcher-level summary.
                        flock -x "${QUEUE_LOCK}" -c "echo '${VID}/${KIND} rc=${RC}' >> '${FAIL_LOG}'"
                        echo "[T2] [GPU ${W}] FAIL  ${VID}/${KIND}  rc=${RC} ($((T1 - T0))s)  log: ${DIAG_LOG}"
                        echo "[T2] [GPU ${W}] tail of ${DIAG_LOG}:"
                        tail -n 20 "${DIAG_LOG}" | sed "s/^/[T2] [GPU ${W}]   /"
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
            echo "[T2] ${N_FAIL}/${N_TASKS} diag tasks FAILED:"
            sed 's/^/[T2]   /' "${FAIL_LOG}"
        else
            echo "[T2] all ${N_TASKS} diag tasks succeeded."
        fi
    fi
fi

# (6) Pack.
if [[ ${DRY_RUN} -eq 0 ]]; then
    STAMP="$(date +%Y%m%d_%H%M%S)"
    PACK="analyses/round29_tier2_results_${STAMP}.tar.gz"
    tar -czf "${PACK}" \
        analyses/round29_tier2_ablation_manifest.* \
        analyses/round29_r29_t2_*_diag_* 2>/dev/null || true
    echo "[T2] Packed ${PACK}"
fi
