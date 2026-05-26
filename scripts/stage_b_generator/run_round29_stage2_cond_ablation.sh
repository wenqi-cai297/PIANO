#!/usr/bin/env bash
# Round-29 Stage-2 condition + injection ablation launcher (Bash).
#
# Mirror of run_round29_stage2_cond_ablation.py for Linux servers. Both
# launchers must stay behavior-equivalent (Codex post-review §P1/§P2).
#
# Usage:
#   bash scripts/stage_b_generator/run_round29_stage2_cond_ablation.sh --group injection
#   bash scripts/stage_b_generator/run_round29_stage2_cond_ablation.sh --group content
#   bash scripts/stage_b_generator/run_round29_stage2_cond_ablation.sh --only r29_a0_input_add
#   bash scripts/stage_b_generator/run_round29_stage2_cond_ablation.sh --group A_injection --dry-run
#
# Group aliases:
#   injection -> A_injection
#   coarse    -> B_coarse
#   interaction -> C_interaction
#   support   -> D_support
#   body      -> E_body
#   final     -> F_final
#   content   -> B_coarse,C_interaction,D_support,E_body
#   all       -> all
#
# Environment overrides:
#   ROUND29_SINGLE_GPU=1                  single-GPU mode (no accelerate launch)
#   ROUND29_NUM_PROCESSES=N               accelerate --num_processes (default:
#                                         nvidia-smi -L count; falls back to 2)
#   ROUND29_PARALLEL_DIAG_WORKERS=N       diag-phase parallel workers (default:
#                                         ROUND29_NUM_PROCESSES; each worker
#                                         pinned to one GPU via CUDA_VISIBLE_DEVICES)
#   ROUND29_DIAG_CKPT_NAME=best_val.pt    diagnostic checkpoint filename
#
# Speedup vs the original sequential single-GPU diag: with N variants
# selected (via --group / --only) and 3 diag kinds per variant, and W
# parallel workers, the diag phase finishes in ceil((3*N) / W)
# sequential durations instead of 3*N. Applies to every group:
#   A (5 variants) → 15 tasks → 5 batches on 3 GPUs (~3× faster)
#   B (7 variants) → 21 tasks → 7 batches
#   E (8 variants) → 24 tasks → 8 batches
#   all (36)       → 108 tasks → 36 batches

set -euo pipefail
cd "$(dirname "$0")/../.."

GROUP="all"
ONLY=""
DRY_RUN=0
SKIP_TRAIN=0
SKIP_EVAL=0
SKIP_PREFLIGHT=0
ALLOW_MISSING_DIAG=0
SINGLE_GPU="${ROUND29_SINGLE_GPU:-0}"
# Auto-detect GPU count via `nvidia-smi -L`. Override with --num-processes
# or ROUND29_NUM_PROCESSES env. Falls back to 2 if nvidia-smi is missing.
if [[ -n "${ROUND29_NUM_PROCESSES:-}" ]]; then
    NUM_PROCESSES="${ROUND29_NUM_PROCESSES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_PROCESSES="$(nvidia-smi -L | wc -l)"
    [[ "${NUM_PROCESSES}" -lt 1 ]] && NUM_PROCESSES=1
else
    NUM_PROCESSES=2
fi
# Diag parallelism — defaults to NUM_PROCESSES (one diag per GPU).
# Override with --parallel-diag-workers N or ROUND29_PARALLEL_DIAG_WORKERS env.
PARALLEL_DIAG_WORKERS="${ROUND29_PARALLEL_DIAG_WORKERS:-${NUM_PROCESSES}}"
DIAG_CKPT_NAME="${ROUND29_DIAG_CKPT_NAME:-final.pt}"
MANIFEST="analyses/round29_stage2_cond_ablation_manifest.json"
LOG_DIR="runs/round29_stage2_cond_ablation"
mkdir -p "${LOG_DIR}"

PY="${PY:-python}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --group)         GROUP="$2"; shift 2 ;;
        --only)          ONLY="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --skip-train)    SKIP_TRAIN=1; shift ;;
        --skip-eval)     SKIP_EVAL=1; shift ;;
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        --allow-missing-diag-inputs) ALLOW_MISSING_DIAG=1; shift ;;
        --diag-ckpt-name) DIAG_CKPT_NAME="$2"; shift 2 ;;
        --num-processes) NUM_PROCESSES="$2"; shift 2 ;;
        --parallel-diag-workers) PARALLEL_DIAG_WORKERS="$2"; shift 2 ;;
        --single-gpu)    SINGLE_GPU=1; shift ;;
        -h|--help)
            sed -n '1,33p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Resolve group aliases to a comma-separated manifest-group string.
resolve_group() {
    case "$1" in
        injection)   echo "A_injection" ;;
        coarse)      echo "B_coarse" ;;
        interaction) echo "C_interaction" ;;
        support)     echo "D_support" ;;
        body)        echo "E_body" ;;
        final)       echo "F_final" ;;
        content)     echo "B_coarse,C_interaction,D_support,E_body" ;;
        all)         echo "all" ;;
        A_injection|B_coarse|C_interaction|D_support|E_body|F_final) echo "$1" ;;
        *)
            echo "ERROR: unknown --group value '$1'" >&2
            exit 2 ;;
    esac
}

GROUPS_RESOLVED="$(resolve_group "${GROUP}")"

echo "[R29] launcher config: NUM_PROCESSES=${NUM_PROCESSES}  PARALLEL_DIAG_WORKERS=${PARALLEL_DIAG_WORKERS}  SINGLE_GPU=${SINGLE_GPU}  DIAG_CKPT_NAME=${DIAG_CKPT_NAME}"

# Generate manifest if missing.
if [[ ! -f "${MANIFEST}" ]]; then
    echo "[R29] Manifest missing — running config generator..."
    GEN_ARGS=()
    [[ -n "${DATASETS_ROOT:-}" ]] && GEN_ARGS+=(--data-root "${DATASETS_ROOT}")
    "${PY}" scripts/stage_b_generator/round29_make_stage2_cond_ablation_configs.py "${GEN_ARGS[@]}"
fi

# Pick variants via Python helper (reads manifest, applies group filter).
PICK_SCRIPT='
import json, sys
m = json.load(open(sys.argv[1]))
groups = sys.argv[2]
only = sys.argv[3]
want_groups = None if groups == "all" else set(groups.split(","))
want_only = set(only.split(",")) if only else None
for v in m["variants"]:
    if want_only is not None:
        if v["variant_id"] not in want_only: continue
    elif want_groups is not None and v["group"] not in want_groups:
        continue
    print(v["variant_id"], v["group"], v["config_path"], v["output_dir"], v["subset_file"])
'
VARIANTS="$("${PY}" -c "${PICK_SCRIPT}" "${MANIFEST}" "${GROUPS_RESOLVED}" "${ONLY}")"

if [[ -z "${VARIANTS}" ]]; then
    echo "[R29] no variants matched group='${GROUP}' only='${ONLY}'"
    exit 0
fi

echo "[R29] Variants to process:"
echo "${VARIANTS}"

# Preflight.
preflight_fail=0
if [[ ${SKIP_PREFLIGHT} -eq 0 && ${DRY_RUN} -eq 0 ]]; then
    echo "[R29] Preflight..."
    while IFS=' ' read -r VID GRP CFG OUTDIR SUBSET; do
        [[ -z "${VID}" ]] && continue
        for p in "${CFG}" "${SUBSET}"; do
            if [[ ! -e "${p}" ]]; then
                echo "    [${VID}] missing: ${p}"
                preflight_fail=1
            fi
        done
        # Selection JSON must carry non-empty {subset, seq_id} pairs.
        if [[ -e "${SUBSET}" && ${SKIP_EVAL} -eq 0 ]]; then
            N_SEL="$("${PY}" -c "
import json, sys
data = json.load(open(sys.argv[1]))
sel = data.get('selected') or data.get('candidates') or data.get('clips') or []
print(len(sel))
" "${SUBSET}")"
            if [[ "${N_SEL}" == "0" ]]; then
                echo "    [${VID}] selection JSON has no {subset, seq_id} pairs: ${SUBSET}"
                preflight_fail=1
            fi
        fi
        # Dataset roots — parse the YAML and verify each root exists.
        # Skipping this lets training fail later at FileNotFoundError.
        if [[ ${SKIP_TRAIN} -eq 0 && -e "${CFG}" ]]; then
            BAD_ROOTS="$("${PY}" -c "
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(open(sys.argv[1]))
for ds in (cfg.get('data', {}).get('datasets') or []):
    root = ds.get('root', '')
    if root and not Path(root).exists():
        print(f\"{ds.get('name')}={root}\")
" "${CFG}")"
            if [[ -n "${BAD_ROOTS}" ]]; then
                while IFS= read -r br; do
                    echo "    [${VID}] dataset root not on disk: ${br}"
                done <<< "${BAD_ROOTS}"
                echo "    [${VID}]   -> re-run generator with --data-root <correct path> or export DATASETS_ROOT=..."
                preflight_fail=1
            fi
        fi
        if [[ ${SKIP_TRAIN} -eq 1 && ${SKIP_EVAL} -eq 0 ]]; then
            DIAG_CKPT="${OUTDIR}/${DIAG_CKPT_NAME}"
            if [[ ! -e "${DIAG_CKPT}" && ${ALLOW_MISSING_DIAG} -eq 0 ]]; then
                echo "    [${VID}] diag ckpt missing: ${DIAG_CKPT}"
                preflight_fail=1
            fi
        fi
    done <<< "${VARIANTS}"
    if [[ ${preflight_fail} -ne 0 ]]; then
        echo "[R29] FATAL preflight failures. Fix them or pass --skip-preflight."
        exit 1
    fi
fi

# Smoke test (fast, dry-run mode).
if [[ ${DRY_RUN} -eq 0 ]]; then
    echo "[R29] Smoke test..."
    "${PY}" scripts/stage_b_generator/round29_stage2_cond_smoke_test.py --dry-run
fi

# Resolve bucket per selection JSON.
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

# =============================================================================
# PHASE 1: training — sequential (accelerate uses all GPUs per variant)
# =============================================================================
TRAINED_OK=""   # accumulate "VID GRP CFG OUTDIR SUBSET" lines, one per success
while IFS=' ' read -r VID GRP CFG OUTDIR SUBSET; do
    [[ -z "${VID}" ]] && continue
    LOG="${LOG_DIR}/${VID}.log"

    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] TRAIN ${VID}  group=${GRP}"
    echo "    config:  ${CFG}"
    echo "    output:  ${OUTDIR}"
    echo "    log:     ${LOG}"
    echo "================================================================"

    # When ${NUM_PROCESSES} == 1, fall through to the single-GPU path so we
    # don't pay the accelerate-launch overhead with no actual distribution.
    if [[ ${SKIP_TRAIN} -eq 0 ]]; then
        if [[ "${SINGLE_GPU}" == "1" || "${NUM_PROCESSES}" == "1" ]]; then
            TRAIN_CMD=("${PY}" -u src/piano/training/train_anchordiff.py --config "${CFG}")
        else
            TRAIN_CMD=(accelerate launch
                --num_processes "${NUM_PROCESSES}" --multi_gpu --mixed_precision bf16
                src/piano/training/train_anchordiff.py --config "${CFG}")
        fi
        if [[ ${DRY_RUN} -eq 1 ]]; then
            echo "[R29 DRY-RUN ${VID} TRAIN]  (NUM_PROCESSES=${NUM_PROCESSES})"
            echo "    \$ ${TRAIN_CMD[*]}"
            TRAINED_OK="${TRAINED_OK}${VID} ${GRP} ${CFG} ${OUTDIR} ${SUBSET}"$'\n'
        else
            if "${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG}"; then
                TRAINED_OK="${TRAINED_OK}${VID} ${GRP} ${CFG} ${OUTDIR} ${SUBSET}"$'\n'
            else
                echo "[R29] WARN: training failed for ${VID}; skipping diag"
            fi
        fi
    else
        echo "--skip-train: skipping training for ${VID}"
        TRAINED_OK="${TRAINED_OK}${VID} ${GRP} ${CFG} ${OUTDIR} ${SUBSET}"$'\n'
    fi
done <<< "${VARIANTS}"

# =============================================================================
# PHASE 2: diagnostics — parallel via CUDA_VISIBLE_DEVICES on N workers.
# Each variant emits 3 diag tasks (sustained_contact / gait / body_action),
# so 5 variants × 3 = 15 tasks; with 3 workers ≈ 3× speedup.
# =============================================================================
if [[ ${SKIP_EVAL} -eq 1 ]]; then
    echo
    echo "--skip-eval: skipping diag for all variants"
elif [[ -z "${TRAINED_OK}" ]]; then
    echo "[R29] No variants succeeded training; no diag tasks to run."
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] DIAG PHASE  (parallel, workers=${PARALLEL_DIAG_WORKERS})"
    echo "================================================================"

    # Build the diag-task queue (one line per (vid, kind, cmd) triple).
    # Using a temp file because bash arrays-of-arrays are clumsy.
    TASK_QUEUE="$(mktemp -t r29_diag_tasks.XXXXXX)"
    trap "rm -f '${TASK_QUEUE}'" EXIT

    while IFS=' ' read -r VID GRP CFG OUTDIR SUBSET; do
        [[ -z "${VID}" ]] && continue
        BUCKET="$(selection_bucket "${SUBSET}")"
        CKPT_PATH="${OUTDIR}/${DIAG_CKPT_NAME}"
        # Pre-flight: ckpt must exist (unless allowed missing).
        if [[ ! -e "${CKPT_PATH}" ]]; then
            if [[ ${ALLOW_MISSING_DIAG} -eq 1 ]]; then
                echo "[R29] WARN: diag ckpt missing for ${VID}: ${CKPT_PATH} (skipped)"
                continue
            fi
            echo "[R29] FATAL: diag ckpt missing for ${VID}: ${CKPT_PATH}"
            echo "      Pass --allow-missing-diag-inputs to skip."
            exit 2
        fi
        for kind in sustained_contact gait body_action; do
            case "${kind}" in
                sustained_contact) DIAG_SCRIPT="scripts/stage_b_generator/round26_sustained_contact_diag.py" ;;
                gait)              DIAG_SCRIPT="scripts/stage_b_generator/round26_gait_diag.py" ;;
                body_action)       DIAG_SCRIPT="scripts/stage_b_generator/round28_body_action_diag.py" ;;
            esac
            OUT_DIR="analyses/round29_${VID}_diag_${kind}"
            mkdir -p "${OUT_DIR}"
            # Tab-separated so spaces in args (none here) wouldn't bite.
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${VID}" "${kind}" "${DIAG_SCRIPT}" "${CFG}" "${CKPT_PATH}" "${SUBSET}" "${BUCKET}" \
                >> "${TASK_QUEUE}"
        done
    done <<< "${TRAINED_OK}"

    N_TASKS="$(wc -l < "${TASK_QUEUE}")"
    echo "[R29] ${N_TASKS} diag tasks queued; launching ${PARALLEL_DIAG_WORKERS} GPU workers..."

    if [[ ${DRY_RUN} -eq 1 ]]; then
        # Just print what would run, round-robin across GPUs.
        IDX=0
        while IFS=$'\t' read -r VID KIND DIAG_SCRIPT CFG CKPT_PATH SUBSET BUCKET; do
            GPU=$((IDX % PARALLEL_DIAG_WORKERS))
            OUT_DIR="analyses/round29_${VID}_diag_${KIND}"
            echo "[R29 DRY-RUN [GPU ${GPU}] ${VID} DIAG/${KIND}]"
            echo "    \$ CUDA_VISIBLE_DEVICES=${GPU} ${PY} -u ${DIAG_SCRIPT} --config ${CFG} --ckpt ${CKPT_PATH} --selection-json ${SUBSET} --output-dir ${OUT_DIR} --bucket ${BUCKET}"
            IDX=$((IDX + 1))
        done < "${TASK_QUEUE}"
        echo "[R29 DRY-RUN] ${N_TASKS} diag tasks across ${PARALLEL_DIAG_WORKERS} GPUs"
    else
        # Real parallel run: per-GPU worker subshell pulls from the queue via flock.
        QUEUE_LOCK="${TASK_QUEUE}.lock"
        : > "${QUEUE_LOCK}"
        WORKER_PIDS=()
        for ((W = 0; W < PARALLEL_DIAG_WORKERS; W++)); do
            (
                while true; do
                    # Atomically take the first remaining line.
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
                    echo "[R29] [GPU ${W}] START ${VID}/${KIND}  log: ${DIAG_LOG}"
                    CUDA_VISIBLE_DEVICES="${W}" \
                        "${PY}" -u "${DIAG_SCRIPT}" \
                        --config "${CFG}" \
                        --ckpt "${CKPT_PATH}" \
                        --selection-json "${SUBSET}" \
                        --output-dir "${OUT_DIR}" \
                        --bucket "${BUCKET}" \
                        >> "${DIAG_LOG}" 2>&1
                    RC=$?
                    T1=$(date +%s)
                    if [[ ${RC} -eq 0 ]]; then
                        echo "[R29] [GPU ${W}] DONE  ${VID}/${KIND}  ($((T1 - T0))s)"
                    else
                        echo "[R29] [GPU ${W}] FAIL  ${VID}/${KIND}  rc=${RC} ($((T1 - T0))s)  log: ${DIAG_LOG}"
                    fi
                done
            ) &
            WORKER_PIDS+=($!)
        done
        # Wait for all workers; collect any failure.
        DIAG_FAILED=0
        for pid in "${WORKER_PIDS[@]}"; do
            if ! wait "${pid}"; then
                DIAG_FAILED=1
            fi
        done
        rm -f "${QUEUE_LOCK}"
        if [[ ${DIAG_FAILED} -ne 0 ]]; then
            echo "[R29] parallel diag: some tasks failed (see per-task logs above)."
        else
            echo "[R29] parallel diag: all tasks succeeded."
        fi
    fi
fi

# Summarize.
if [[ ${DRY_RUN} -eq 0 ]]; then
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] Summarizing..."
    "${PY}" -u scripts/stage_b_generator/round29_summarize_stage2_cond_ablation.py \
        --manifest "${MANIFEST}" \
        --output-json analyses/round29_stage2_cond_ablation_summary.json \
        --output-md   analyses/round29_stage2_cond_ablation_summary.md \
        --allow-missing-results \
        2>&1 | tee -a "${LOG_DIR}/summary.log"
fi

# Pack.
if [[ ${DRY_RUN} -eq 0 ]]; then
    STAMP="$(date +%Y%m%d_%H%M%S)"
    PACK="analyses/round29_results_${STAMP}.tar.gz"
    tar -czf "${PACK}" \
        analyses/round29_stage2_cond_ablation_manifest.* \
        analyses/round29_stage2_cond_ablation_summary.* \
        analyses/round29_*_diag_* 2>/dev/null || true
    echo "[R29] Packed ${PACK}"
fi
