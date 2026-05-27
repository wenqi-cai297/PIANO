#!/usr/bin/env bash
# Round-29 next-step ablations launcher (A0/A1/H1/A2).
#
# Per analyses/2026-05-28_round29_next_step_ablation_execution_prompt_for_claude_code.md.
#
# 4 new train variants + R0/B1/G1 references (not retrained, but their
# existing diag stats are required) + invalid old H1 (historical only):
#   r29_ns_a0_c41_g1_loss_s4     C41 + G1 losses (S4 in data, loss-only)
#   r29_ns_a1_c41_s4_g1          C41 + S4 consumed + G1 losses
#   r29_ns_h1_i5_upper_bound     R0 cond with I3 swapped for I5
#   r29_ns_a2_c41_i5_g1          C41 + I5 + G1 losses
#
# Schedule: bs=32 / accum=1 / 80 ep / heldout val / val_every=5 /
# save_every=10 / warmup=250 (2× 5080).
#
# Phase 1 (TRAIN):  4 sequential variants, each on all GPUs via accelerate.
# Phase 2 (DIAG):   4 × 4 kinds (sustained_contact, gait, body_action,
#                                g1_soft_stance) × 2 buckets +
#                   1 motion-repr-floor diag per bucket (no ckpt).
#                   G1 soft-stance only runs on variants with G1 losses
#                   (A0/A1/A2). H1 has no G1 losses so it skips that diag.
# Phase 3 (SUMM):   reads R0/B1/G1 ref stats + new diag dirs + repr floor;
#                   writes analyses/2026-05-28_round29_next_step_ablation_report.md.
# Phase 4 (PACK):   tarballs everything (including ref-stats copies for
#                   reproducibility) into analyses/round29_next_step_ablation_results_<stamp>.tar.gz.
#
# Usage:
#   bash scripts/stage_b_generator/run_round29_next_step_ablations.sh
#   bash scripts/stage_b_generator/run_round29_next_step_ablations.sh --dry-run
#   bash scripts/stage_b_generator/run_round29_next_step_ablations.sh \
#       --only r29_ns_a0_c41_g1_loss_s4,r29_ns_h1_i5_upper_bound
#   bash scripts/stage_b_generator/run_round29_next_step_ablations.sh --skip-train
#
# Environment overrides:
#   DATASETS_ROOT=...                       dataset root (default = dev Windows path)
#   ROUND29_NS_NUM_PROCESSES=N              accelerate --num_processes
#   ROUND29_NS_PARALLEL_DIAG_WORKERS=N      diag workers (default: NUM_PROCESSES)
#   ROUND29_NS_DIAG_CKPT_NAME=best_val.pt   diag ckpt filename (default: final.pt)
#   ROUND29_NS_SINGLE_GPU=1                 force single-GPU train
#   ROUND29_NS_ALLOW_PARTIAL=1              allow partial reports on failures
#   ROUND29_NS_REGEN_CONFIGS=1              force manifest/config regeneration

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
SKIP_TRAIN=0
SKIP_EVAL=0
SINGLE_GPU="${ROUND29_NS_SINGLE_GPU:-0}"
DIAG_CKPT_NAME="${ROUND29_NS_DIAG_CKPT_NAME:-final.pt}"
ALLOW_PARTIAL="${ROUND29_NS_ALLOW_PARTIAL:-0}"
REGEN_CONFIGS="${ROUND29_NS_REGEN_CONFIGS:-0}"

SELECTION_TRAIN="analyses/round27_tier0_train_indices_48_balanced.json"
SELECTION_VAL="analyses/round29_val_diag_indices_48_balanced.json"

# Reference diag dirs (required by summarizer).
R0_REF_VARIANT="r29_ft_r0_clean_a3_baseline"
B1_REF_VARIANT="r29_nb_b1_c41_only"
G1_REF_VARIANT="r29_nb_g1_phasefree_gait_fixed"
# G1 reference config + ckpt — required to generate the G1 soft-stance
# reference diag that the summarizer needs as the gait anchor for A0/A1/A2.
G1_REF_CFG="configs/training/anchordiff_${G1_REF_VARIANT}.yaml"
G1_REF_CKPT="runs/training/stageB_anchordiff_${G1_REF_VARIANT}/${DIAG_CKPT_NAME}"
# Invalid old H1 — must NOT be used as a valid contact-content reference.
OLD_H1_VARIANT="r29_nb_h1_r0_plus_oracle_full_hint"

# Variants that ship with G1 losses (and therefore need the soft-stance diag).
G1_LOSS_VARIANTS=(
    "r29_ns_a0_c41_g1_loss_s4"
    "r29_ns_a1_c41_s4_g1"
    "r29_ns_a2_c41_i5_g1"
)

if [[ -n "${ROUND29_NS_NUM_PROCESSES:-}" ]]; then
    NUM_PROCESSES="${ROUND29_NS_NUM_PROCESSES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_PROCESSES="$(nvidia-smi -L | wc -l)"
    [[ "${NUM_PROCESSES}" -lt 1 ]] && NUM_PROCESSES=1
else
    NUM_PROCESSES=2
fi
PARALLEL_DIAG_WORKERS="${ROUND29_NS_PARALLEL_DIAG_WORKERS:-${NUM_PROCESSES}}"

MANIFEST="analyses/round29_next_step_ablation_manifest.json"
LOG_DIR="runs/round29_next_step_ablation"
mkdir -p "${LOG_DIR}"
if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PY="python"
    elif command -v python3 >/dev/null 2>&1; then
        PY="python3"
    else
        echo "[NS] FATAL: neither python nor python3 was found" >&2
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
        --regen-configs)         REGEN_CONFIGS=1; shift ;;
        -h|--help)
            sed -n '1,45p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

echo "[NS] NUM_PROCESSES=${NUM_PROCESSES}  PARALLEL_DIAG_WORKERS=${PARALLEL_DIAG_WORKERS}  DIAG_CKPT_NAME=${DIAG_CKPT_NAME}  ALLOW_PARTIAL=${ALLOW_PARTIAL}"

# (1) Generate manifest + configs if missing or stale.
GENERATOR="scripts/stage_b_generator/round29_make_next_step_ablation_configs.py"
if [[ ${REGEN_CONFIGS} -eq 1 || ! -f "${MANIFEST}" || "${GENERATOR}" -nt "${MANIFEST}" || -n "${DATASETS_ROOT:-}" ]]; then
    echo "[NS] Regenerating manifest/configs..."
    GEN_ARGS=()
    [[ -n "${DATASETS_ROOT:-}" ]] && GEN_ARGS+=(--data-root "${DATASETS_ROOT}")
    "${PY}" "${GENERATOR}" "${GEN_ARGS[@]}"
fi

# (2) Pick TRAIN-able variants from manifest (skip reference rows).
PICK_SCRIPT='
import json, sys
m = json.load(open(sys.argv[1]))
only = sys.argv[2]
want_only = set(only.split(",")) if only else None
for v in m["variants"]:
    if not v.get("train", True): continue
    if want_only is not None and v["variant_id"] not in want_only: continue
    print(v["variant_id"], v["config_path"], v["output_dir"])
'
VARIANTS="$("${PY}" -c "${PICK_SCRIPT}" "${MANIFEST}" "${ONLY}")"
if [[ -z "${VARIANTS}" ]]; then
    echo "[NS] no train variants matched only='${ONLY}'"
    exit 0
fi
echo "[NS] Train variants to process:"
echo "${VARIANTS}"

# (3) Preflight — split into "always run" and "not-dry-run only" checks.
# Per Codex review (2026-05-28 fix prompt §4): dry-run must catch missing
# launch prerequisites before the user starts a long tmux job.
preflight_fail=0

# --- Always-run checks (in dry-run too) ---
# Train configs exist + no dead oracle_hint fields anywhere.
while IFS=' ' read -r VID CFG OUTDIR; do
    [[ -z "${VID}" ]] && continue
    if [[ ! -e "${CFG}" ]]; then
        echo "[NS PREFLIGHT FAIL] [${VID}] missing config: ${CFG}"
        preflight_fail=1
    elif grep -q "oracle_hint" "${CFG}" 2>/dev/null; then
        echo "[NS PREFLIGHT FAIL] [${VID}] config contains dead oracle_hint fields: ${CFG}"
        preflight_fail=1
    fi
done <<< "${VARIANTS}"

# Selection JSONs + reference diag dirs + G1 ref config — required by the
# diag phase and the summarizer; check in dry-run too so a doomed launch
# is caught before the user starts a long tmux job.
if [[ ${SKIP_EVAL} -eq 0 ]]; then
    for sel in "${SELECTION_TRAIN}" "${SELECTION_VAL}"; do
        if [[ ! -e "${sel}" ]]; then
            echo "[NS PREFLIGHT FAIL] missing diag selection JSON: ${sel}"
            preflight_fail=1
        fi
    done
    # Reference diag dirs for R0/B1/G1 (standard 6 dirs each = 3 kinds × 2 buckets).
    REF_COUNT=0
    for ref_vid in "${R0_REF_VARIANT}" "${B1_REF_VARIANT}" "${G1_REF_VARIANT}"; do
        REF_DIRS="$(ls -d analyses/round29_${ref_vid}_diag_sustained_contact_train analyses/round29_${ref_vid}_diag_sustained_contact_val analyses/round29_${ref_vid}_diag_gait_train analyses/round29_${ref_vid}_diag_gait_val analyses/round29_${ref_vid}_diag_body_action_train analyses/round29_${ref_vid}_diag_body_action_val 2>/dev/null | wc -l)"
        if [[ "${REF_DIRS}" -lt 6 ]]; then
            echo "[NS PREFLIGHT FAIL] missing standard reference diag dirs for ${ref_vid} (need 6 {sustained_contact,gait,body_action} × {train,val}, found ${REF_DIRS})"
            echo "    -> generate via the matching matrix's launcher with --only ${ref_vid} --skip-train"
            preflight_fail=1
        else
            REF_COUNT=$((REF_COUNT + REF_DIRS))
        fi
    done
    # G1 reference config required because we queue a G1 soft-stance diag
    # against it (so the summarizer's G1 soft-stance row is non-empty).
    if [[ ! -e "${G1_REF_CFG}" ]]; then
        echo "[NS PREFLIGHT FAIL] G1 reference config missing: ${G1_REF_CFG}"
        echo "    -> regenerate via the previous matrix:"
        echo "       python scripts/stage_b_generator/round29_make_next_ablation_configs.py"
        preflight_fail=1
    fi
    # Sanity: invalid old H1 — allowed on disk, just warn.
    OLD_H1_DIRS="$(ls -d analyses/round29_${OLD_H1_VARIANT}_diag_* 2>/dev/null | wc -l)"
    if [[ "${OLD_H1_DIRS}" -gt 0 ]]; then
        echo "[NS] note: ${OLD_H1_DIRS} old H1 diag dirs present; summarizer will mark them INVALID (never used as decision reference)."
    fi
    echo "[NS] standard reference diag dirs total: ${REF_COUNT}"
fi

# --- Not-dry-run-only checks ---
if [[ ${DRY_RUN} -eq 0 ]]; then
    # Dataset roots only matter if we will train.
    if [[ ${SKIP_TRAIN} -eq 0 ]]; then
        while IFS=' ' read -r VID CFG OUTDIR; do
            [[ -z "${VID}" ]] && continue
            [[ ! -e "${CFG}" ]] && continue
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
                    echo "[NS PREFLIGHT FAIL] [${VID}] dataset root not on disk: ${br}"
                done <<< "${BAD}"
                echo "    [${VID}]   -> re-run generator with --data-root <correct path> or export DATASETS_ROOT=..."
                preflight_fail=1
            fi
        done <<< "${VARIANTS}"
    fi
    # G1 reference ckpt — needed because we queue the G1 soft-stance diag
    # against it. WARN under ALLOW_PARTIAL=1, FATAL otherwise.
    if [[ ${SKIP_EVAL} -eq 0 && ! -e "${G1_REF_CKPT}" ]]; then
        if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
            echo "[NS PREFLIGHT WARN] G1 reference ckpt missing (ALLOW_PARTIAL=1): ${G1_REF_CKPT}"
        else
            echo "[NS PREFLIGHT FAIL] G1 reference ckpt missing: ${G1_REF_CKPT}"
            echo "    -> server should already have it from the previous matrix; if not,"
            echo "       regenerate via:"
            echo "       bash scripts/stage_b_generator/run_round29_next_ablations.sh --only ${G1_REF_VARIANT}"
            preflight_fail=1
        fi
    fi
else
    # Dry-run: print informational WARN if G1 ckpt is missing but don't fail.
    if [[ ${SKIP_EVAL} -eq 0 && ! -e "${G1_REF_CKPT}" ]]; then
        echo "[NS PREFLIGHT WARN] G1 reference ckpt missing (dry-run): ${G1_REF_CKPT}"
        echo "    (a real launch would FATAL unless ALLOW_PARTIAL=1)"
    fi
fi

if [[ ${preflight_fail} -ne 0 ]]; then
    echo "[NS] FATAL preflight failures."
    exit 1
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
            echo "[NS DRY-RUN ${VID} TRAIN]"
            echo "    \$ ${TRAIN_CMD[*]}"
            TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR}"$'\n'
        else
            if "${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG}"; then
                TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR}"$'\n'
            else
                if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
                    echo "[NS] WARN: training failed for ${VID}; skipping diag"
                else
                    echo "[NS] FATAL: training failed for ${VID}; aborting full matrix."
                    exit 1
                fi
            fi
        fi
    else
        echo "--skip-train: skipping training for ${VID}"
        TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR}"$'\n'
    fi
done <<< "${VARIANTS}"

# (5) PHASE 2: DIAG parallel.
#     Per variant: 3 standard kinds × 2 buckets = 6 tasks.
#     For G1-loss variants (A0/A1/A2): + g1_soft_stance × 2 buckets = +2.
#     Plus: motion-repr-floor diag × 2 buckets (no ckpt; needs a config).
if [[ ${SKIP_EVAL} -eq 1 ]]; then
    echo
    echo "--skip-eval: skipping diag"
elif [[ -z "${TRAINED_OK}" ]]; then
    echo "[NS] No variants succeeded training; no diag to run."
    if [[ "${ALLOW_PARTIAL}" != "1" ]]; then
        exit 1
    fi
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] DIAG PHASE (workers=${PARALLEL_DIAG_WORKERS})"
    echo "================================================================"

    TASK_QUEUE="$(mktemp -t ns_diag_tasks.XXXXXX)"
    QUEUE_LOCK="${TASK_QUEUE}.lock"
    FAIL_LOG="${TASK_QUEUE}.fail"
    : > "${QUEUE_LOCK}"
    : > "${FAIL_LOG}"
    trap "rm -f '${TASK_QUEUE}' '${QUEUE_LOCK}' '${FAIL_LOG}'" EXIT

    # Pick one of the train variants' config for the repr-floor diag
    # (any of them works — repr floor doesn't depend on the model).
    REPR_FLOOR_CFG=""
    while IFS=' ' read -r VID CFG OUTDIR; do
        [[ -z "${VID}" ]] && continue
        REPR_FLOOR_CFG="${CFG}"
        break
    done <<< "${TRAINED_OK}"

    while IFS=' ' read -r VID CFG OUTDIR; do
        [[ -z "${VID}" ]] && continue
        CKPT_PATH="${OUTDIR}/${DIAG_CKPT_NAME}"
        if [[ ! -e "${CKPT_PATH}" && ${DRY_RUN} -eq 0 ]]; then
            if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
                echo "[NS] WARN: diag ckpt missing: ${CKPT_PATH} (skipped)"
                continue
            fi
            echo "[NS] FATAL: diag ckpt missing: ${CKPT_PATH}"
            exit 1
        fi
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
        # G1 soft-stance: only for variants that ship with G1 losses.
        for G1V in "${G1_LOSS_VARIANTS[@]}"; do
            if [[ "${VID}" == "${G1V}" ]]; then
                for sublabel in train val; do
                    case "${sublabel}" in
                        train) SUBSET_PATH="${SELECTION_TRAIN}" ;;
                        val)   SUBSET_PATH="${SELECTION_VAL}" ;;
                    esac
                    OUT_DIR="analyses/round29_${VID}_diag_g1_soft_stance_${sublabel}"
                    mkdir -p "${OUT_DIR}"
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                        "${VID}" "g1_soft_stance_${sublabel}" \
                        "scripts/stage_b_generator/round29_g1_soft_stance_diag.py" \
                        "${CFG}" "${CKPT_PATH}" "${SUBSET_PATH}" "${sublabel}" \
                        >> "${TASK_QUEUE}"
                done
                break
            fi
        done
    done <<< "${TRAINED_OK}"

    # G1 reference soft-stance — required by summarizer as the gait anchor
    # for A0/A1/A2. We use the existing G1 ref config + ckpt (NOT retraining).
    # Skip if (a) ALLOW_PARTIAL=1 AND ckpt missing, or (b) the diag stats
    # already exist on disk. Otherwise queue both buckets.
    if [[ -e "${G1_REF_CFG}" ]]; then
        G1_REF_CKPT_AVAIL=1
        if [[ ! -e "${G1_REF_CKPT}" && ${DRY_RUN} -eq 0 ]]; then
            if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
                echo "[NS] WARN: G1 reference ckpt missing (ALLOW_PARTIAL=1); skipping G1 soft-stance diag"
                G1_REF_CKPT_AVAIL=0
            else
                echo "[NS] FATAL: G1 reference ckpt missing: ${G1_REF_CKPT}"
                exit 1
            fi
        fi
        if [[ ${G1_REF_CKPT_AVAIL} -eq 1 ]]; then
            for sublabel in train val; do
                case "${sublabel}" in
                    train) SUBSET_PATH="${SELECTION_TRAIN}" ;;
                    val)   SUBSET_PATH="${SELECTION_VAL}" ;;
                esac
                OUT_DIR="analyses/round29_${G1_REF_VARIANT}_diag_g1_soft_stance_${sublabel}"
                # Skip if stats already exist on disk — we don't have to
                # rerun a 48-clip diag against the same ckpt.
                if [[ -e "${OUT_DIR}/g1_soft_stance_stats.json" && ${DRY_RUN} -eq 0 ]]; then
                    echo "[NS] G1 ref soft-stance ${sublabel}: stats exist, reusing"
                    continue
                fi
                mkdir -p "${OUT_DIR}"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "${G1_REF_VARIANT}" "g1_soft_stance_${sublabel}" \
                    "scripts/stage_b_generator/round29_g1_soft_stance_diag.py" \
                    "${G1_REF_CFG}" "${G1_REF_CKPT}" "${SUBSET_PATH}" "${sublabel}" \
                    >> "${TASK_QUEUE}"
            done
        fi
    fi

    # Motion-repr floor (no ckpt). One task per bucket.
    if [[ -n "${REPR_FLOOR_CFG}" ]]; then
        for sublabel in train val; do
            case "${sublabel}" in
                train) SUBSET_PATH="${SELECTION_TRAIN}" ;;
                val)   SUBSET_PATH="${SELECTION_VAL}" ;;
            esac
            OUT_DIR="analyses/round29_repr_floor_${sublabel}"
            mkdir -p "${OUT_DIR}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "_repr_floor_" "repr_floor_${sublabel}" \
                "scripts/stage_b_generator/round29_motion_repr_roundtrip_diag.py" \
                "${REPR_FLOOR_CFG}" "_no_ckpt_" "${SUBSET_PATH}" "${sublabel}" \
                >> "${TASK_QUEUE}"
        done
    fi

    N_TASKS="$(wc -l < "${TASK_QUEUE}")"
    if [[ "${N_TASKS}" -eq 0 && ${DRY_RUN} -eq 0 ]]; then
        echo "[NS] FATAL: no diag tasks were queued."
        exit 1
    fi
    echo "[NS] ${N_TASKS} diag tasks queued; launching ${PARALLEL_DIAG_WORKERS} GPU workers..."

    if [[ ${DRY_RUN} -eq 1 ]]; then
        IDX=0
        while IFS=$'\t' read -r VID KIND DIAG_SCRIPT CFG CKPT_PATH SUBSET BUCKET; do
            GPU=$((IDX % PARALLEL_DIAG_WORKERS))
            if [[ "${VID}" == "_repr_floor_" ]]; then
                OUT_DIR="analyses/round29_repr_floor_${BUCKET}"
                echo "[NS DRY-RUN [GPU ${GPU}] REPR_FLOOR ${BUCKET}]"
                echo "    \$ CUDA_VISIBLE_DEVICES=${GPU} ${PY} -u ${DIAG_SCRIPT} --config ${CFG} --selection-json ${SUBSET} --output-dir ${OUT_DIR} --bucket ${BUCKET}"
            else
                OUT_DIR="analyses/round29_${VID}_diag_${KIND}"
                echo "[NS DRY-RUN [GPU ${GPU}] ${VID} DIAG/${KIND}]"
                echo "    \$ CUDA_VISIBLE_DEVICES=${GPU} ${PY} -u ${DIAG_SCRIPT} --config ${CFG} --ckpt ${CKPT_PATH} --selection-json ${SUBSET} --output-dir ${OUT_DIR} --bucket ${BUCKET}"
            fi
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
                    if [[ "${VID}" == "_repr_floor_" ]]; then
                        OUT_DIR="analyses/round29_repr_floor_${BUCKET}"
                        DIAG_LOG="${LOG_DIR}/repr_floor_${BUCKET}.log"
                        DIAG_CMD=("${PY}" -u "${DIAG_SCRIPT}"
                            --config "${CFG}"
                            --selection-json "${SUBSET}"
                            --output-dir "${OUT_DIR}"
                            --bucket "${BUCKET}")
                    else
                        OUT_DIR="analyses/round29_${VID}_diag_${KIND}"
                        DIAG_LOG="${LOG_DIR}/${VID}_diag_${KIND}.log"
                        DIAG_CMD=("${PY}" -u "${DIAG_SCRIPT}"
                            --config "${CFG}"
                            --ckpt "${CKPT_PATH}"
                            --selection-json "${SUBSET}"
                            --output-dir "${OUT_DIR}"
                            --bucket "${BUCKET}")
                    fi
                    T0=$(date +%s)
                    echo "[NS] [GPU ${W}] START ${VID}/${KIND}  log: ${DIAG_LOG}"
                    : > "${DIAG_LOG}"
                    set +e
                    CUDA_VISIBLE_DEVICES="${W}" \
                        "${DIAG_CMD[@]}" > "${DIAG_LOG}" 2>&1
                    RC=$?
                    set -e
                    T1=$(date +%s)
                    if [[ ${RC} -eq 0 ]]; then
                        echo "[NS] [GPU ${W}] DONE  ${VID}/${KIND}  ($((T1 - T0))s)"
                    else
                        flock -x "${QUEUE_LOCK}" -c "echo '${VID}/${KIND} rc=${RC}' >> '${FAIL_LOG}'"
                        echo "[NS] [GPU ${W}] FAIL  ${VID}/${KIND}  rc=${RC} ($((T1 - T0))s)  log: ${DIAG_LOG}"
                        echo "[NS] [GPU ${W}] tail of ${DIAG_LOG}:"
                        tail -n 20 "${DIAG_LOG}" | sed "s/^/[NS] [GPU ${W}]   /"
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
            echo "[NS] ${N_FAIL}/${N_TASKS} diag tasks FAILED:"
            sed 's/^/[NS]   /' "${FAIL_LOG}"
            if [[ "${ALLOW_PARTIAL}" != "1" ]]; then
                echo "[NS] FATAL: diag failures are not allowed in the full matrix."
                exit 1
            fi
        else
            echo "[NS] all ${N_TASKS} diag tasks succeeded."
        fi
    fi
fi

# (6) PHASE 3: SUMMARIZE.
SUMMARY_MD="analyses/2026-05-28_round29_next_step_ablation_report.md"
if [[ ${DRY_RUN} -eq 1 ]]; then
    echo
    echo "[NS DRY-RUN] would run summarizer:"
    echo "    \$ ${PY} scripts/stage_b_generator/round29_summarize_next_step_ablation.py --out ${SUMMARY_MD}"
elif [[ ${SKIP_EVAL} -eq 0 ]]; then
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] SUMMARIZE -> ${SUMMARY_MD}"
    echo "================================================================"
    SUMM_ARGS=(--out "${SUMMARY_MD}")
    [[ "${ALLOW_PARTIAL}" == "1" ]] && SUMM_ARGS+=(--allow-partial)
    if ! "${PY}" scripts/stage_b_generator/round29_summarize_next_step_ablation.py \
        "${SUMM_ARGS[@]}"; then
        if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
            echo "[NS] WARN: summarizer failed; report may be partial"
        else
            echo "[NS] FATAL: summarizer failed."
            exit 1
        fi
    fi
fi

# (7) PHASE 4: PACK.
if [[ ${DRY_RUN} -eq 0 ]]; then
    STAMP="$(date +%Y%m%d_%H%M%S)"
    PACK="analyses/round29_next_step_ablation_results_${STAMP}.tar.gz"
    PACK_ITEMS=(
        analyses/round29_next_step_ablation_manifest.*
        configs/training/anchordiff_r29_ns_*.yaml
    )
    if [[ ${SKIP_EVAL} -eq 0 ]]; then
        PACK_ITEMS+=(
            analyses/round29_r29_ns_*_diag_*
            analyses/round29_repr_floor_*
            "${SUMMARY_MD}"
        )
    fi
    # Reference stats (so the tarball can be diff'd against R0/B1/G1
    # without needing the upstream matrices on the same machine).
    REF_COPY_ROOT="analyses/round29_next_step_reference_stats"
    mkdir -p "${REF_COPY_ROOT}"
    for ref_vid in "${R0_REF_VARIANT}" "${B1_REF_VARIANT}" "${G1_REF_VARIANT}"; do
        for ref_dir in analyses/round29_${ref_vid}_diag_*; do
            if [[ -d "${ref_dir}" ]]; then
                dest="${REF_COPY_ROOT}/$(basename "${ref_dir}")"
                if [[ ! -d "${dest}" ]]; then
                    cp -r "${ref_dir}" "${dest}"
                fi
            fi
        done
    done
    PACK_ITEMS+=("${REF_COPY_ROOT}")
    if ! tar -czf "${PACK}" "${PACK_ITEMS[@]}"; then
        if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
            echo "[NS] WARN: result pack failed: ${PACK}"
        else
            echo "[NS] FATAL: result pack failed: ${PACK}"
            exit 1
        fi
    fi
    echo "[NS] Packed ${PACK}"
fi
