#!/usr/bin/env bash
# Round-29 loss-strategy FULL-DATA launcher (v2 winner promotion).
#
# Per Codex review of v2 48-clip result:
#   analyses/2026-05-27_round29_loss_strategy_v2_codex_review.md §Final recommendation
#
# 4-run full-data matrix:
#   r29_lsf_a2_baseline_from_scratch  A2 (adapter_only), original a-group losses
#   r29_lsf_a3_baseline_from_scratch  A3 (input_add_adapter), original a-group losses
#   r29_lsf_a2_anchor2_mixed          A2, v2 winner loss weights
#   r29_lsf_a3_anchor2_mixed          A3, v2 winner loss weights
#
# Schedule: 80 ep, full InterAct train set (no subset_indices_file),
# heldout val (val_every=5), save_every=10, warmup=250.
#
# Phase 1 (TRAIN): 4 variants sequential, each uses all GPUs via accelerate.
# Phase 2 (DIAG): 4 variants × 3 diag kinds × 2 subsets = 24 tasks parallel
#                 across N GPU workers. Two subsets:
#   - train: 48-clip balanced (same as A-group / v2 48-clip diag)
#            for in-distribution sanity + cross-experiment comparability.
#   - val:   48-clip heldout-val balanced (built by
#            round29_build_val_diag_subset.py) for generalization measurement.
# Phase 3 (SUMMARIZE): writes
#   analyses/2026-05-27_round29_loss_strategy_full_data_report.md
# Phase 4 (PACK): tarballs manifest + diag + report into
#   analyses/round29_loss_strategy_full_data_results_<stamp>.tar.gz
#
# Usage:
#   bash scripts/stage_b_generator/run_round29_loss_strategy_full_data.sh
#   bash scripts/stage_b_generator/run_round29_loss_strategy_full_data.sh --dry-run
#   bash scripts/stage_b_generator/run_round29_loss_strategy_full_data.sh \
#       --only r29_lsf_a2_anchor2_mixed,r29_lsf_a3_anchor2_mixed
#   bash scripts/stage_b_generator/run_round29_loss_strategy_full_data.sh --skip-train
#
# Environment overrides:
#   DATASETS_ROOT=...                  dataset root (default = dev Windows path)
#   ROUND29_NUM_PROCESSES=N            accelerate --num_processes (default: nvidia-smi -L)
#   ROUND29_PARALLEL_DIAG_WORKERS=N    diag workers (default: NUM_PROCESSES)
#   ROUND29_DIAG_CKPT_NAME=best_val.pt diag ckpt filename (default: final.pt)
#   ROUND29_SINGLE_GPU=1               force single-GPU train

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
SKIP_TRAIN=0
SKIP_EVAL=0
SINGLE_GPU="${ROUND29_SINGLE_GPU:-0}"
DIAG_CKPT_NAME="${ROUND29_DIAG_CKPT_NAME:-final.pt}"
# Diag subsets — same 48-clip selections the loss-strategy 48-clip
# ablation used so cross-protocol comparisons (48-clip mechanism screen
# vs full-data) are directly possible.
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

MANIFEST="analyses/round29_loss_strategy_full_data_manifest.json"
LOG_DIR="runs/round29_loss_strategy_full_data"
mkdir -p "${LOG_DIR}"
if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PY="python"
    elif command -v python3 >/dev/null 2>&1; then
        PY="python3"
    else
        echo "[LSF] FATAL: neither python nor python3 was found; set PY=/path/to/python" >&2
        exit 127
    fi
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)                  ONLY="$2"; shift 2 ;;
        --dry-run)               DRY_RUN=1; shift ;;
        --skip-train)            SKIP_TRAIN=1; shift ;;
        --skip-eval)             SKIP_EVAL=1; shift ;;
        --diag-ckpt-name)        DIAG_CKPT_NAME="$2"; shift 2 ;;
        --num-processes)         NUM_PROCESSES="$2"; shift 2 ;;
        --parallel-diag-workers) PARALLEL_DIAG_WORKERS="$2"; shift 2 ;;
        --single-gpu)            SINGLE_GPU=1; shift ;;
        -h|--help)
            sed -n '1,40p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

echo "[LSF] NUM_PROCESSES=${NUM_PROCESSES}  PARALLEL_DIAG_WORKERS=${PARALLEL_DIAG_WORKERS}  DIAG_CKPT_NAME=${DIAG_CKPT_NAME}"

# (1) Generate manifest + configs if missing or stale.
if [[ ! -f "${MANIFEST}" ]]; then
    echo "[LSF] Manifest missing — running config generator..."
    GEN_ARGS=()
    [[ -n "${DATASETS_ROOT:-}" ]] && GEN_ARGS+=(--data-root "${DATASETS_ROOT}")
    "${PY}" scripts/stage_b_generator/round29_make_loss_strategy_full_data_configs.py "${GEN_ARGS[@]}"
fi

# (2) Pick variants from manifest.
PICK_SCRIPT='
import json, sys
m = json.load(open(sys.argv[1]))
only = sys.argv[2]
want_only = set(only.split(",")) if only else None
for v in m["variants"]:
    if want_only is not None and v["variant_id"] not in want_only: continue
    print(v["variant_id"], v["config_path"], v["output_dir"])
'
VARIANTS="$("${PY}" -c "${PICK_SCRIPT}" "${MANIFEST}" "${ONLY}")"

if [[ -z "${VARIANTS}" ]]; then
    echo "[LSF] no variants matched only='${ONLY}'"
    exit 0
fi

echo "[LSF] Variants to process:"
echo "${VARIANTS}"

# (3) Preflight.
if [[ ${DRY_RUN} -eq 0 ]]; then
    preflight_fail=0
    while IFS=' ' read -r VID CFG OUTDIR; do
        [[ -z "${VID}" ]] && continue
        if [[ ! -e "${CFG}" ]]; then
            echo "    [${VID}] missing config: ${CFG}"
            preflight_fail=1
        fi
        # Dataset roots — parse YAML to verify on-disk.
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
    # Diag selection files — needed for Phase 2.
    if [[ ${SKIP_EVAL} -eq 0 ]]; then
        for sel in "${SELECTION_TRAIN}" "${SELECTION_VAL}"; do
            if [[ ! -e "${sel}" ]]; then
                echo "    missing diag selection JSON: ${sel}"
                if [[ "${sel}" == "${SELECTION_VAL}" ]]; then
                    echo "    -> generate it with: python scripts/stage_b_generator/round29_build_val_diag_subset.py --config configs/training/anchordiff_r29_lsf_a2_baseline_from_scratch.yaml"
                fi
                preflight_fail=1
            fi
        done
    fi
    if [[ ${preflight_fail} -ne 0 ]]; then
        echo "[LSF] FATAL preflight failures."
        exit 1
    fi
fi

# (4) PHASE 1: TRAIN sequentially.
TRAINED_OK=""
while IFS=' ' read -r VID CFG OUTDIR; do
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
            echo "[LSF DRY-RUN ${VID} TRAIN]"
            echo "    \$ ${TRAIN_CMD[*]}"
            TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR}"$'\n'
        else
            if "${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG}"; then
                TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR}"$'\n'
            else
                echo "[LSF] WARN: training failed for ${VID}; skipping diag"
            fi
        fi
    else
        echo "--skip-train: skipping training for ${VID}"
        TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR}"$'\n'
    fi
done <<< "${VARIANTS}"

# (5) PHASE 2: DIAG parallel across GPUs (3 diag kinds × 2 subsets per variant).
if [[ ${SKIP_EVAL} -eq 1 ]]; then
    echo
    echo "--skip-eval: skipping diag"
elif [[ -z "${TRAINED_OK}" ]]; then
    echo "[LSF] No variants succeeded training; no diag to run."
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] DIAG PHASE (workers=${PARALLEL_DIAG_WORKERS})"
    echo "================================================================"

    TASK_QUEUE="$(mktemp -t lsf_diag_tasks.XXXXXX)"
    QUEUE_LOCK="${TASK_QUEUE}.lock"
    FAIL_LOG="${TASK_QUEUE}.fail"
    : > "${QUEUE_LOCK}"
    : > "${FAIL_LOG}"
    trap "rm -f '${TASK_QUEUE}' '${QUEUE_LOCK}' '${FAIL_LOG}'" EXIT

    while IFS=' ' read -r VID CFG OUTDIR; do
        [[ -z "${VID}" ]] && continue
        CKPT_PATH="${OUTDIR}/${DIAG_CKPT_NAME}"
        if [[ ! -e "${CKPT_PATH}" && ${DRY_RUN} -eq 0 ]]; then
            echo "[LSF] WARN: diag ckpt missing: ${CKPT_PATH} (skipped)"
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
                OUT_DIR="analyses/round29_${VID}_diag_${kind}_${sublabel}"
                mkdir -p "${OUT_DIR}"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "${VID}" "${kind}_${sublabel}" "${DIAG_SCRIPT}" "${CFG}" "${CKPT_PATH}" "${SUBSET_PATH}" "${sublabel}" \
                    >> "${TASK_QUEUE}"
            done
        done
    done <<< "${TRAINED_OK}"

    N_TASKS="$(wc -l < "${TASK_QUEUE}")"
    echo "[LSF] ${N_TASKS} diag tasks queued; launching ${PARALLEL_DIAG_WORKERS} GPU workers..."

    if [[ ${DRY_RUN} -eq 1 ]]; then
        IDX=0
        while IFS=$'\t' read -r VID KIND DIAG_SCRIPT CFG CKPT_PATH SUBSET BUCKET; do
            GPU=$((IDX % PARALLEL_DIAG_WORKERS))
            OUT_DIR="analyses/round29_${VID}_diag_${KIND}"
            echo "[LSF DRY-RUN [GPU ${GPU}] ${VID} DIAG/${KIND}]"
            echo "    \$ CUDA_VISIBLE_DEVICES=${GPU} ${PY} -u ${DIAG_SCRIPT} --config ${CFG} --ckpt ${CKPT_PATH} --selection-json ${SUBSET} --output-dir ${OUT_DIR} --bucket ${BUCKET}"
            IDX=$((IDX + 1))
        done < "${TASK_QUEUE}"
    else
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
                    echo "[LSF] [GPU ${W}] START ${VID}/${KIND}  log: ${DIAG_LOG}"
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
                        echo "[LSF] [GPU ${W}] DONE  ${VID}/${KIND}  ($((T1 - T0))s)"
                    else
                        flock -x "${QUEUE_LOCK}" -c "echo '${VID}/${KIND} rc=${RC}' >> '${FAIL_LOG}'"
                        echo "[LSF] [GPU ${W}] FAIL  ${VID}/${KIND}  rc=${RC} ($((T1 - T0))s)  log: ${DIAG_LOG}"
                        echo "[LSF] [GPU ${W}] tail of ${DIAG_LOG}:"
                        tail -n 20 "${DIAG_LOG}" | sed "s/^/[LSF] [GPU ${W}]   /"
                    fi
                done
            ) &
            WORKER_PIDS+=($!)
        done
        for pid in "${WORKER_PIDS[@]}"; do
            wait "${pid}" || true
        done
        N_FAIL=$(wc -l < "${FAIL_LOG}" 2>/dev/null || echo 0)
        if [[ ${N_FAIL} -gt 0 ]]; then
            echo "[LSF] ${N_FAIL}/${N_TASKS} diag tasks FAILED:"
            sed 's/^/[LSF]   /' "${FAIL_LOG}"
        else
            echo "[LSF] all ${N_TASKS} diag tasks succeeded."
        fi
    fi
fi

# (6) PHASE 3: SUMMARIZE — write the 4-variant comparison MD.
SUMMARY_MD="analyses/2026-05-27_round29_loss_strategy_full_data_report.md"
if [[ ${DRY_RUN} -eq 1 ]]; then
    echo
    echo "[LSF DRY-RUN] would run summarizer:"
    echo "    \$ ${PY} scripts/stage_b_generator/round29_summarize_loss_strategy_full_data.py --out ${SUMMARY_MD}"
elif [[ ${SKIP_EVAL} -eq 0 ]]; then
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] SUMMARIZE -> ${SUMMARY_MD}"
    echo "================================================================"
    "${PY}" scripts/stage_b_generator/round29_summarize_loss_strategy_full_data.py \
        --out "${SUMMARY_MD}" || echo "[LSF] WARN: summarizer failed; report may be partial"
fi

# (7) PHASE 4: PACK — bundle manifest + diag + summary report.
if [[ ${DRY_RUN} -eq 0 ]]; then
    STAMP="$(date +%Y%m%d_%H%M%S)"
    PACK="analyses/round29_loss_strategy_full_data_results_${STAMP}.tar.gz"
    tar -czf "${PACK}" \
        analyses/round29_loss_strategy_full_data_manifest.* \
        analyses/round29_r29_lsf_*_diag_* \
        "${SUMMARY_MD}" 2>/dev/null || true
    echo "[LSF] Packed ${PACK}"
fi
