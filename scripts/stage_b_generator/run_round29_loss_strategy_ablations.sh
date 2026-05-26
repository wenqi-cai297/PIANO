#!/usr/bin/env bash
# Round-29 loss-strategy ablation launcher.
#
# Per analyses/2026-05-27_round29_loss_strategy_ablation_prompt_for_claude_code.md.
#
# Six v2 variants on the 48-clip balanced subset (300 ep), per Codex review
# (analyses/2026-05-27_round29_loss_strategy_codex_review.md).
# Three families × A2/A3 injection:
#   r29_ls_{a2,a3}_baseline_from_scratch    fair from-scratch baseline
#                                            (no init_checkpoint, original
#                                             a-group losses) — provides the
#                                             missing Rule-1 reference.
#   r29_ls_{a2,a3}_relbeh_v2_anchor0_low    pure low-weight condition strat
#                                            (anchor=0, R29 weights 0.10).
#   r29_ls_{a2,a3}_relbeh_v2_anchor2_mixed  weak absolute stabilizer
#                                            (anchor_pos=2, anchor_vel=0.5)
#                                            + low R29 weights.
# All v2 families include swing_clearance (Codex P0+ patch) at weight 0.10
# with 5 cm threshold to fight the "both feet planted" minimum v1 produced.
#
# Phase 1 (TRAIN): 6 variants sequential, each uses all GPUs via accelerate.
# Phase 2 (DIAG): 6 variants × 3 diag = 18 tasks parallel across N GPU workers.
#
# Diag uses the same 48-clip balanced subset as the A-group baselines
# so deltas are directly comparable to a2/a3 final.pt results.
#
# Usage:
#   bash scripts/stage_b_generator/run_round29_loss_strategy_ablations.sh
#   bash scripts/stage_b_generator/run_round29_loss_strategy_ablations.sh --dry-run
#   bash scripts/stage_b_generator/run_round29_loss_strategy_ablations.sh \\
#       --only r29_ls_a2_relbeh_v2_anchor0_low,r29_ls_a3_relbeh_v2_anchor0_low
#   bash scripts/stage_b_generator/run_round29_loss_strategy_ablations.sh --skip-train
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

if [[ -n "${ROUND29_NUM_PROCESSES:-}" ]]; then
    NUM_PROCESSES="${ROUND29_NUM_PROCESSES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_PROCESSES="$(nvidia-smi -L | wc -l)"
    [[ "${NUM_PROCESSES}" -lt 1 ]] && NUM_PROCESSES=1
else
    NUM_PROCESSES=2
fi
PARALLEL_DIAG_WORKERS="${ROUND29_PARALLEL_DIAG_WORKERS:-${NUM_PROCESSES}}"

MANIFEST="analyses/round29_loss_strategy_ablation_manifest.json"
LOG_DIR="runs/round29_loss_strategy_ablation"
mkdir -p "${LOG_DIR}"
if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PY="python"
    elif command -v python3 >/dev/null 2>&1; then
        PY="python3"
    else
        echo "[LS] FATAL: neither python nor python3 was found; set PY=/path/to/python" >&2
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
            sed -n '1,35p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

echo "[LS] NUM_PROCESSES=${NUM_PROCESSES}  PARALLEL_DIAG_WORKERS=${PARALLEL_DIAG_WORKERS}  DIAG_CKPT_NAME=${DIAG_CKPT_NAME}"

# (1) Generate manifest + configs if missing or stale.
if [[ ! -f "${MANIFEST}" ]]; then
    echo "[LS] Manifest missing — running config generator..."
    GEN_ARGS=()
    [[ -n "${DATASETS_ROOT:-}" ]] && GEN_ARGS+=(--data-root "${DATASETS_ROOT}")
    "${PY}" scripts/stage_b_generator/round29_make_loss_strategy_ablation_configs.py "${GEN_ARGS[@]}"
fi

# (2) Pick variants from manifest.
PICK_SCRIPT='
import json, sys
m = json.load(open(sys.argv[1]))
only = sys.argv[2]
want_only = set(only.split(",")) if only else None
for v in m["variants"]:
    if want_only is not None and v["variant_id"] not in want_only: continue
    print(v["variant_id"], v["config_path"], v["output_dir"], v["subset_file"])
'
VARIANTS="$("${PY}" -c "${PICK_SCRIPT}" "${MANIFEST}" "${ONLY}")"

if [[ -z "${VARIANTS}" ]]; then
    echo "[LS] no variants matched only='${ONLY}'"
    exit 0
fi

echo "[LS] Variants to process:"
echo "${VARIANTS}"

# (3) Preflight.
if [[ ${DRY_RUN} -eq 0 ]]; then
    preflight_fail=0
    while IFS=' ' read -r VID CFG OUTDIR SUBSET; do
        [[ -z "${VID}" ]] && continue
        for p in "${CFG}" "${SUBSET}"; do
            if [[ ! -e "${p}" ]]; then
                echo "    [${VID}] missing: ${p}"
                preflight_fail=1
            fi
        done
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
        echo "[LS] FATAL preflight failures."
        exit 1
    fi
fi

# (4) PHASE 1: TRAIN sequentially.
TRAINED_OK=""
while IFS=' ' read -r VID CFG OUTDIR SUBSET; do
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
            echo "[LS DRY-RUN ${VID} TRAIN]"
            echo "    \$ ${TRAIN_CMD[*]}"
            TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR} ${SUBSET}"$'\n'
        else
            if "${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG}"; then
                TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR} ${SUBSET}"$'\n'
            else
                echo "[LS] WARN: training failed for ${VID}; skipping diag"
            fi
        fi
    else
        echo "--skip-train: skipping training for ${VID}"
        TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR} ${SUBSET}"$'\n'
    fi
done <<< "${VARIANTS}"

# (5) PHASE 2: DIAG parallel across GPUs.
if [[ ${SKIP_EVAL} -eq 1 ]]; then
    echo
    echo "--skip-eval: skipping diag"
elif [[ -z "${TRAINED_OK}" ]]; then
    echo "[LS] No variants succeeded training; no diag to run."
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] DIAG PHASE (workers=${PARALLEL_DIAG_WORKERS})"
    echo "================================================================"

    TASK_QUEUE="$(mktemp -t ls_diag_tasks.XXXXXX)"
    FAIL_LOG="${TASK_QUEUE}.fail"
    QUEUE_LOCK="${TASK_QUEUE}.lock"
    : > "${QUEUE_LOCK}"
    : > "${FAIL_LOG}"
    trap "rm -f '${TASK_QUEUE}' '${QUEUE_LOCK}' '${FAIL_LOG}'" EXIT

    selection_bucket() {
        local sel="$1"
        if [[ ! -e "${sel}" ]]; then echo "train"; return; fi
        "${PY}" -c "
import json, sys
data = json.load(open(sys.argv[1]))
b = data.get('bucket', 'train')
if b not in ('train','val'): b = 'train'
print(b)" "${sel}"
    }

    while IFS=' ' read -r VID CFG OUTDIR SUBSET; do
        [[ -z "${VID}" ]] && continue
        BUCKET="$(selection_bucket "${SUBSET}")"
        CKPT_PATH="${OUTDIR}/${DIAG_CKPT_NAME}"
        if [[ ! -e "${CKPT_PATH}" && ${DRY_RUN} -eq 0 ]]; then
            echo "[LS] WARN: diag ckpt missing: ${CKPT_PATH} (skipped)"
            continue
        fi
        for kind in sustained_contact gait body_action; do
            case "${kind}" in
                sustained_contact) DIAG_SCRIPT="scripts/stage_b_generator/round26_sustained_contact_diag.py" ;;
                gait)              DIAG_SCRIPT="scripts/stage_b_generator/round26_gait_diag.py" ;;
                body_action)       DIAG_SCRIPT="scripts/stage_b_generator/round28_body_action_diag.py" ;;
            esac
            OUT_DIR="analyses/round29_${VID}_diag_${kind}"
            mkdir -p "${OUT_DIR}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${VID}" "${kind}" "${DIAG_SCRIPT}" "${CFG}" "${CKPT_PATH}" "${SUBSET}" "${BUCKET}" \
                >> "${TASK_QUEUE}"
        done
    done <<< "${TRAINED_OK}"

    N_TASKS="$(wc -l < "${TASK_QUEUE}")"
    echo "[LS] ${N_TASKS} diag tasks queued; launching ${PARALLEL_DIAG_WORKERS} GPU workers..."

    if [[ ${DRY_RUN} -eq 1 ]]; then
        IDX=0
        while IFS=$'\t' read -r VID KIND DIAG_SCRIPT CFG CKPT_PATH SUBSET BUCKET; do
            GPU=$((IDX % PARALLEL_DIAG_WORKERS))
            OUT_DIR="analyses/round29_${VID}_diag_${KIND}"
            echo "[LS DRY-RUN [GPU ${GPU}] ${VID} DIAG/${KIND}]"
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
                    echo "[LS] [GPU ${W}] START ${VID}/${KIND}  log: ${DIAG_LOG}"
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
                        echo "[LS] [GPU ${W}] DONE  ${VID}/${KIND}  ($((T1 - T0))s)"
                    else
                        flock -x "${QUEUE_LOCK}" -c "echo '${VID}/${KIND} rc=${RC}' >> '${FAIL_LOG}'"
                        echo "[LS] [GPU ${W}] FAIL  ${VID}/${KIND}  rc=${RC} ($((T1 - T0))s)  log: ${DIAG_LOG}"
                        echo "[LS] [GPU ${W}] tail of ${DIAG_LOG}:"
                        tail -n 20 "${DIAG_LOG}" | sed "s/^/[LS] [GPU ${W}]   /"
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
            echo "[LS] ${N_FAIL}/${N_TASKS} diag tasks FAILED:"
            sed 's/^/[LS]   /' "${FAIL_LOG}"
        else
            echo "[LS] all ${N_TASKS} diag tasks succeeded."
        fi
    fi
fi

# (6) Summarize.
# Per analyses/2026-05-27_round29_loss_strategy_ablation_prompt_for_claude_code.md §9.3:
# emit a comparison Markdown report against the A2/A3 a-group baselines.
SUMMARY_MD="analyses/2026-05-27_round29_loss_strategy_ablation_report.md"
if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "[LS DRY-RUN] would run summarizer:"
    echo "    \$ ${PY} scripts/stage_b_generator/round29_summarize_loss_strategy_results.py --out ${SUMMARY_MD}"
elif [[ ${SKIP_EVAL} -eq 0 ]]; then
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] SUMMARIZE -> ${SUMMARY_MD}"
    echo "================================================================"
    "${PY}" scripts/stage_b_generator/round29_summarize_loss_strategy_results.py \
        --out "${SUMMARY_MD}" || echo "[LS] WARN: summarizer failed; report may be partial"
fi

# (7) Pack — bundle manifests + all diag outputs + summary report into one
# tarball the user can download in a single transfer.
if [[ ${DRY_RUN} -eq 0 ]]; then
    STAMP="$(date +%Y%m%d_%H%M%S)"
    PACK="analyses/round29_loss_strategy_results_${STAMP}.tar.gz"
    tar -czf "${PACK}" \
        analyses/round29_loss_strategy_ablation_manifest.* \
        analyses/round29_r29_ls_*_diag_* \
        "${SUMMARY_MD}" 2>/dev/null || true
    echo "[LS] Packed ${PACK}"
fi
