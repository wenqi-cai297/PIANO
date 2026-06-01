#!/usr/bin/env bash
# Round-41 Stage-1 cascade fine-tune matrix.
#
# 5 cells = A0 control + A1-A4 cascade ablation. All cells fine-tune
# from V8 V6 ckpt (drift_max 17.43 cm direct, 16.81 cm via cascade with
# GT C41/S4) under frozen R29 PB1 ckpt.
#
# Per-cell calibration phase: smoke-test each cell, log cascade vs
# stage1_self loss ratio. If ratio > 3.0, abort + warn (user must
# manually scale cascade.w_total down). This guards against R36-style
# scale-dominate disasters before any 40-epoch training starts.
#
# Usage:
#   tmux new -s r41
#   export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
#   bash scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh
#
#   bash ... --dry-run
#   bash ... --only stage1_r41_a0_cascade_off,stage1_r41_a1_motion_mse
#
# Env overrides:
#   ROUND41_GPUS="0,2"
#   ROUND41_NUM_PROCESSES=N            accelerate --num_processes
#   ROUND41_BUCKETS="val"              diag bucket
#   ROUND41_BASE_CFG=...               base V8 V6 cfg
#   ROUND41_PB1_CKPT=...               PB1 ship ckpt
#   ROUND41_SKIP_CALIBRATION=1         skip per-cell smoke probe
#   ROUND41_ABORT_IF_CASCADE_RATIO_OVER=3.0
#   ROUND41_ALLOW_PARTIAL=1            keep going on per-variant failure

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
SKIP_TRAIN=0
SKIP_DIAG=0
SKIP_CALIBRATION="${ROUND41_SKIP_CALIBRATION:-0}"
FORCE_RETRAIN=0
FORCE_REDIAG=0
ALLOW_PARTIAL="${ROUND41_ALLOW_PARTIAL:-0}"
BUCKETS_STR="${ROUND41_BUCKETS:-val}"
ABORT_RATIO="${ROUND41_ABORT_IF_CASCADE_RATIO_OVER:-3.0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)              ONLY="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=1; shift ;;
        --skip-train)        SKIP_TRAIN=1; shift ;;
        --skip-diag)         SKIP_DIAG=1; shift ;;
        --skip-calibration)  SKIP_CALIBRATION=1; shift ;;
        --force-retrain)     FORCE_RETRAIN=1; shift ;;
        --force-rediag)      FORCE_REDIAG=1; shift ;;
        --buckets)           BUCKETS_STR="$2"; shift 2 ;;
        -h|--help)           sed -n '1,40p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${DATASETS_ROOT:-}" ]]; then
    echo "[R41] FATAL: export DATASETS_ROOT before launch." >&2
    exit 1
fi

GPUS="${ROUND41_GPUS:-0,2}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
NUM_GPUS_IN_MASK="$(echo "${GPUS}" | tr ',' '\n' | grep -c '^[0-9]\+$' || true)"
if [[ "${NUM_GPUS_IN_MASK}" -lt 1 ]]; then NUM_GPUS_IN_MASK=1; fi
: "${ROUND41_NUM_PROCESSES:=${NUM_GPUS_IN_MASK}}"
export ROUND41_NUM_PROCESSES

ROUND41_BASE_CFG="${ROUND41_BASE_CFG:-configs/training/stage1_v8_v6_full_f1.yaml}"
PB1_VARIANT="r29_pb_a1_adaln_s4"
PB1_CKPT="${ROUND41_PB1_CKPT:-runs/training/stageB_anchordiff_${PB1_VARIANT}/final.pt}"
SELECTION_VAL="analyses/round29_val_diag_indices_48_balanced.json"

OVERALL_LOG_DIR="runs/round41_cascade_matrix"
mkdir -p "${OVERALL_LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${OVERALL_LOG_DIR}/summary_${STAMP}.log"

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python";
    elif command -v python3 >/dev/null 2>&1; then PY="python3";
    else echo "[R41] FATAL: no python found" >&2; exit 127; fi
fi

log() { echo "$@" | tee -a "${SUMMARY_LOG}"; }

# ─── Regenerate R41 configs ──────────────────────────────────────────
if [[ ${DRY_RUN} -eq 0 ]]; then
    log "[R41] Regenerating R41 configs from ${ROUND41_BASE_CFG}."
    "${PY}" scripts/stage_a_generator/round41_make_stage1_cascade_configs.py \
        --base-cfg "${ROUND41_BASE_CFG}" \
        --out-dir configs/training/ 2>&1 | tee -a "${SUMMARY_LOG}"
fi

# Hard-coded variant table — matches round41_make_stage1_cascade_configs.py.
VARIANT_TABLE=(
    "stage1_r41_a0_cascade_off   configs/training/stage1_r41_a0_cascade_off.yaml   runs/training/stage1_r41_a0_cascade_off"
    "stage1_r41_a1_motion_mse    configs/training/stage1_r41_a1_motion_mse.yaml    runs/training/stage1_r41_a1_motion_mse"
    "stage1_r41_a2_world_vel     configs/training/stage1_r41_a2_world_vel.yaml     runs/training/stage1_r41_a2_world_vel"
    "stage1_r41_a3_l_pos_full    configs/training/stage1_r41_a3_l_pos_full.yaml    runs/training/stage1_r41_a3_l_pos_full"
    "stage1_r41_a4_anchor_pos    configs/training/stage1_r41_a4_anchor_pos.yaml    runs/training/stage1_r41_a4_anchor_pos"
)

VARIANTS=""
for row in "${VARIANT_TABLE[@]}"; do
    VID="$(echo "${row}" | awk '{print $1}')"
    if [[ -n "${ONLY}" ]]; then
        case ",${ONLY}," in *",${VID},"*) ;; *) continue ;; esac
    fi
    VARIANTS+="${row}"$'\n'
done

log
log "===== R41 cascade matrix launch ${STAMP} ====="
log "*** R36/R37 GUARDRAIL: per-cell smoke test BEFORE training ***"
log "If any non-control cell shows cascade_weighted/mse_x0 > ${ABORT_RATIO},"
log "the launcher aborts. Manually reduce cascade.w_total in the yaml."
log
log "DATASETS_ROOT=${DATASETS_ROOT}"
log "GPUS=${GPUS}  NUM_PROCESSES=${ROUND41_NUM_PROCESSES}"
log "PB1_CKPT=${PB1_CKPT}"
log "Variants:"
echo "${VARIANTS}" | sed 's/^/  /' | tee -a "${SUMMARY_LOG}"
log

# ─── Preflight ───────────────────────────────────────────────────────
preflight_fail=0
if [[ ${DRY_RUN} -eq 0 ]]; then
    if [[ ! -f "${PB1_CKPT}" ]]; then
        log "[R41 PREFLIGHT FAIL] PB1 ckpt missing: ${PB1_CKPT}"
        preflight_fail=1
    fi
    if [[ ! -f "runs/training/stage1_v8_v6_full_f1/final.pt" ]]; then
        log "[R41 PREFLIGHT FAIL] V8 V6 ckpt missing"
        preflight_fail=1
    fi
fi
if [[ ${preflight_fail} -ne 0 ]]; then
    log "[R41] FATAL preflight failures."
    exit 1
fi

# ─── Per-variant loop ──────────────────────────────────────────────
TRAINED_OK_VIDS=()
DIAGED_OK_VIDS=()

while IFS=' ' read -r VID CFG OUTDIR; do
    [[ -z "${VID}" ]] && continue
    VARIANT_LOG="${OVERALL_LOG_DIR}/${VID}.log"

    # ─── Calibration smoke test ────────────────────────────────────
    if [[ ${SKIP_CALIBRATION} -eq 0 && ${DRY_RUN} -eq 0 ]]; then
        log
        log "================================================================"
        log "[$(date '+%F %T')] CALIBRATION SMOKE ${VID}"
        log "================================================================"
        SMOKE_LOG="${OVERALL_LOG_DIR}/${VID}.smoke.log"
        set +e
        # Smoke test runs single-GPU single-process for clean log; the
        # trainer's --smoke-test path is non-DDP.
        CUDA_VISIBLE_DEVICES="$(echo "${GPUS}" | cut -d',' -f1)" \
            "${PY}" -u src/piano/training/train_stage1.py \
                --config "${CFG}" --smoke-test 2>&1 | tee "${SMOKE_LOG}"
        smoke_rc=${PIPESTATUS[0]}
        set -e
        if [[ ${smoke_rc} -ne 0 ]]; then
            log "[R41] [${VID}] smoke test FAILED (rc=${smoke_rc})"
            if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
            continue
        fi
        # Parse cascade ratio from smoke log if cascade is enabled.
        RATIO_LINE=$(grep "R41 cascade weighted" "${SMOKE_LOG}" 2>/dev/null | tail -1 || true)
        if [[ -n "${RATIO_LINE}" ]]; then
            RATIO_VAL=$(echo "${RATIO_LINE}" | awk -F'weighted/mse_x0=' '{print $2}' | awk '{print $1}')
            log "[R41] [${VID}] cascade weighted/mse_x0 = ${RATIO_VAL}"
            # Use awk for float comparison.
            if [[ -n "${RATIO_VAL}" ]] && awk -v r="${RATIO_VAL}" -v t="${ABORT_RATIO}" \
                'BEGIN {exit !(r > t)}'; then
                log "[R41] [${VID}] FATAL: cascade ratio ${RATIO_VAL} > ${ABORT_RATIO}."
                log "[R41]   Edit ${CFG} and reduce cascade.w_total (try /5)."
                log "[R41]   Then re-run with --only ${VID}."
                if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
                continue
            fi
        fi
    fi

    # ─── Phase 1: TRAIN ─────────────────────────────────────────────
    if [[ ${SKIP_TRAIN} -eq 0 ]]; then
        FINAL="${OUTDIR}/final.pt"
        if [[ -f "${FINAL}" && ${FORCE_RETRAIN} -eq 0 ]]; then
            log "[R41] [${VID}] ckpt already exists; skipping (use --force-retrain)"
            TRAINED_OK_VIDS+=("${VID}")
        else
            log
            log "================================================================"
            log "[$(date '+%F %T')] TRAIN ${VID}"
            log "    cfg: ${CFG}    out: ${OUTDIR}"
            log "    GPUs: ${GPUS}  procs=${ROUND41_NUM_PROCESSES}"
            log "================================================================"
            if [[ "${ROUND41_NUM_PROCESSES}" -le 1 ]]; then
                TRAIN_CMD=("${PY}" -u src/piano/training/train_stage1.py
                    --config "${CFG}")
            else
                TRAIN_CMD=(accelerate launch
                    --num_processes "${ROUND41_NUM_PROCESSES}"
                    --multi_gpu --mixed_precision bf16
                    src/piano/training/train_stage1.py --config "${CFG}")
            fi
            if [[ ${DRY_RUN} -eq 1 ]]; then
                log "[R41 DRY-RUN] would train ${VID}"
                log "    \$ CUDA_VISIBLE_DEVICES=${GPUS} ${TRAIN_CMD[*]}"
                TRAINED_OK_VIDS+=("${VID}")
            else
                set +e
                "${TRAIN_CMD[@]}" 2>&1 | tee "${VARIANT_LOG}"
                rc=${PIPESTATUS[0]}
                set -e
                if [[ ${rc} -eq 0 && -f "${FINAL}" ]]; then
                    TRAINED_OK_VIDS+=("${VID}")
                    log "[R41] [${VID}] TRAIN OK"
                else
                    log "[R41] [${VID}] TRAIN FAILED (rc=${rc})"
                    if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
                    continue
                fi
            fi
        fi
    else
        log "[R41] --skip-train: skipping for ${VID}"
        TRAINED_OK_VIDS+=("${VID}")
    fi

    # ─── Phase 2: Direct diag (Stage-1 → PB1, oracle C41/S4) ───────
    if [[ ${SKIP_DIAG} -eq 0 ]]; then
        FINAL="${OUTDIR}/final.pt"
        if [[ ! -f "${FINAL}" && ${DRY_RUN} -eq 0 ]]; then
            log "[R41] [${VID}] no ckpt to diag; skip"
            continue
        fi
        DIRECT_ARCHIVE="analyses/round41_stage1_direct_diag_${VID}"
        DIAG_DONE_MARKER="${DIRECT_ARCHIVE}/sustained_contact_val/sustained_contact_summary.md"
        if [[ -f "${DIAG_DONE_MARKER}" && ${FORCE_REDIAG} -eq 0 ]]; then
            log "[R41] [${VID}] direct diag already archived; skip"
            DIAGED_OK_VIDS+=("${VID}")
        else
            log
            log "================================================================"
            log "[$(date '+%F %T')] DIRECT DIAG ${VID}  (buckets: ${BUCKETS_STR})"
            log "================================================================"
            DS_OUT_TAG="_r41_${VID}"
            DIAG_DIR_ROOT="analyses/round31_stage1_downstream_diag${DS_OUT_TAG}"
            DS_SUB_DIR_ROOT="analyses/round31_stage1_substitute_conds${DS_OUT_TAG}"
            if [[ ${DRY_RUN} -eq 1 ]]; then
                log "[R41 DRY-RUN] direct diag ${VID}"
                DIAGED_OK_VIDS+=("${VID}")
            else
                rm -rf "${DS_SUB_DIR_ROOT}" "${DIAG_DIR_ROOT}"
                set +e
                ROUND31_DS_STAGE1_CFG="${CFG}" \
                ROUND31_DS_STAGE1_CKPT="${FINAL}" \
                ROUND31_DS_PB1_CKPT="${PB1_CKPT}" \
                ROUND31_DS_STAGE1_CFG_SCALE=1.0 \
                ROUND31_DS_STAGE1_SAMPLER=ddim_eta0 \
                ROUND31_DS_BUCKETS="${BUCKETS_STR}" \
                ROUND31_DS_OUT_TAG="${DS_OUT_TAG}" \
                    bash scripts/stage_a_generator/run_round31_stage1_downstream_diag.sh \
                    2>&1 | tee -a "${VARIANT_LOG}"
                rc=${PIPESTATUS[0]}
                set -e
                if [[ ${rc} -eq 0 && -d "${DIAG_DIR_ROOT}" ]]; then
                    rm -rf "${DIRECT_ARCHIVE}"
                    mv "${DIAG_DIR_ROOT}" "${DIRECT_ARCHIVE}"
                    log "[R41] [${VID}] DIRECT DIAG OK -> ${DIRECT_ARCHIVE}"
                    DIAGED_OK_VIDS+=("${VID}")
                else
                    log "[R41] [${VID}] DIRECT DIAG FAILED (rc=${rc})"
                    if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
                fi
            fi
        fi
    fi
done <<< "${VARIANTS}"

# ─── Summary ────────────────────────────────────────────────────────
log
log "================================================================"
log "[$(date '+%F %T')] R41 COMPLETE"
log "================================================================"
log "Trained: ${TRAINED_OK_VIDS[*]:-none}"
log "Diaged:  ${DIAGED_OK_VIDS[*]:-none}"

# ─── Pack ───────────────────────────────────────────────────────────
if [[ ${DRY_RUN} -eq 0 ]]; then
    TARBALL="analyses/round41_cascade_results_${STAMP}.tar.gz"
    log
    log "================================================================"
    log "[$(date '+%F %T')] PACK -> ${TARBALL}"
    log "================================================================"
    ROUND41_STAMP="${STAMP}" \
    ROUND41_TARBALL="${TARBALL}" \
    ROUND41_SUMMARY_LOG="${SUMMARY_LOG}" \
    ROUND41_VARIANT_LOG_DIR="${OVERALL_LOG_DIR}" \
    ROUND41_TRAINED_VIDS="${TRAINED_OK_VIDS[*]:-}" \
    ROUND41_DIAGED_VIDS="${DIAGED_OK_VIDS[*]:-}" \
        bash scripts/stage_a_generator/pack_round41_cascade_sync.sh \
        2>&1 | tee -a "${SUMMARY_LOG}" || \
        log "[R41 PACK] packer failed (non-fatal)"
fi
