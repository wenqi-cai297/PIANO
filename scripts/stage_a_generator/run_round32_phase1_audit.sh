#!/usr/bin/env bash
# Round-32 Phase 1 — Stage-1.5 V0 failure-mode audit + PB1 downstream
# diag, all in one shot.
#
# Mirrors the R31 Phase 1 audit pipeline (`round31_phase1_dyn_audit.py`
# + `run_round31_phase1_dyn_audit.sh`) for the Stage-1.5 (C41+S4) output.
# The Stage-1.5 V0 ckpt was trained with a trivial loss design (MSE on
# C41 + MSE on S4 + a few BCEs / unit-circle hinges); we expect the
# same mode-collapse failure mode R31 V0 exhibited. This launcher makes
# the failure mode explicit on TWO independent axes:
#
#   1. Distribution-level audit (round32_phase1_dyn_audit.py):
#      per-channel mean/std/vel_rms ratios; PSD band ratios; phase
#      unit-circle violation; binary-channel saturation; per-clip
#      wrist drift + frame-0 invariant check.
#
#   2. Downstream coupling (run_round32_stage1p5_downstream_diag.sh):
#      pipe Stage-1.5 V0 sample as C41+S4 cond into frozen PB1 (oracle
#      Stage-1 cond left intact), run the 4 standard diag kinds. Tells
#      us how much V0's imperfect output degrades PB1's wrist drift_max
#      independently of Stage-1. Cross-checks the audit's verdict.
#
# Phases:
#   1) Stage-1.5 V0 sample → cache C41+S4 per clip in val (uses the
#      existing DS32 launcher with --skip-diag so we only get the cache).
#   2) Run PB1 downstream diag with that cache as substitute_conds (DS32
#      launcher with --skip-sample so we reuse step-1 cache).
#   3) Run the distribution-level audit on the same cache.
#   4) Pack everything in one tarball.
#
# GPU restriction: defaults to CUDA_VISIBLE_DEVICES=0,2 (matches V7/V8).
# Total time: ~10 min sample + ~40 min 4 diags + ~5 min audit = ~55 min.
#
# Usage:
#   tmux new -s r32p1
#   export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
#   bash scripts/stage_a_generator/run_round32_phase1_audit.sh
#
#   # Audit only (skip downstream diag), ~15 min:
#   bash scripts/stage_a_generator/run_round32_phase1_audit.sh --skip-downstream
#
#   # Resume after a partial — sample cache and/or downstream output
#   # already on disk are detected and reused.
#
# Environment overrides:
#   ROUND32_P1_GPUS="0,2"          CUDA_VISIBLE_DEVICES (default 0,2)
#   ROUND32_P1_STAGE1P5_CFG=...    Stage-1.5 train cfg path
#   ROUND32_P1_STAGE1P5_CKPT=...   Stage-1.5 ckpt path
#   ROUND32_P1_PB1_CKPT=...        PB1 ckpt path
#   ROUND32_P1_SELECTION=...       val selection json
#   ROUND32_P1_BUCKETS="val"       diag buckets (default val only)

set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
SKIP_SAMPLE=0
SKIP_DOWNSTREAM=0
SKIP_AUDIT=0
FORCE_RESAMPLE=0
FORCE_REDIAG=0
FORCE_REAUDIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)         DRY_RUN=1; shift ;;
        --skip-sample)     SKIP_SAMPLE=1; shift ;;
        --skip-downstream) SKIP_DOWNSTREAM=1; shift ;;
        --skip-audit)      SKIP_AUDIT=1; shift ;;
        --force-resample)  FORCE_RESAMPLE=1; shift ;;
        --force-rediag)    FORCE_REDIAG=1; shift ;;
        --force-reaudit)   FORCE_REAUDIT=1; shift ;;
        -h|--help)         sed -n '1,55p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${DATASETS_ROOT:-}" ]]; then
    echo "[P32] FATAL: export DATASETS_ROOT before launch." >&2
    echo "    export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4" >&2
    exit 1
fi

GPUS="${ROUND32_P1_GPUS:-0,2}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

STAGE1P5_CFG="${ROUND32_P1_STAGE1P5_CFG:-configs/training/stage1p5_interaction_v0.yaml}"
STAGE1P5_CKPT="${ROUND32_P1_STAGE1P5_CKPT:-runs/training/stage1p5_interaction_v0/final.pt}"
PB1_VARIANT="r29_pb_a1_adaln_s4"
PB1_CFG="configs/training/anchordiff_${PB1_VARIANT}.yaml"
PB1_CKPT="${ROUND32_P1_PB1_CKPT:-runs/training/stageB_anchordiff_${PB1_VARIANT}/final.pt}"
SELECTION_VAL="${ROUND32_P1_SELECTION:-analyses/round29_val_diag_indices_48_balanced.json}"
BUCKETS_STR="${ROUND32_P1_BUCKETS:-val}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_TAG="_p1_audit"

# Where the DS32 launcher will write sample cache and diag output (we
# pass the same OUT_TAG so the downstream-diag step reuses our cache).
SUB_DIR_ROOT="analyses/round32_stage1p5_substitute_conds${OUT_TAG}"
DIAG_DIR_ROOT="analyses/round32_stage1p5_downstream_diag${OUT_TAG}"
LOG_DIR="runs/round32_p1_audit"
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/summary_${STAMP}.log"

AUDIT_OUT_DIR="analyses/round32_phase1_dyn_audit_${STAMP}"

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python";
    elif command -v python3 >/dev/null 2>&1; then PY="python3";
    else echo "[P32] FATAL: neither python nor python3 was found" >&2; exit 127; fi
fi

log() { echo "$@" | tee -a "${SUMMARY_LOG}"; }

# Preflight (skipped under --dry-run).
preflight_fail=0
if [[ ${DRY_RUN} -eq 0 ]]; then
    for p in "${STAGE1P5_CFG}" "${STAGE1P5_CKPT}" "${SELECTION_VAL}"; do
        [[ ! -e "${p}" ]] && { log "[P32 PREFLIGHT FAIL] missing: ${p}"; preflight_fail=1; }
    done
    if [[ ${SKIP_DOWNSTREAM} -eq 0 ]]; then
        [[ ! -e "${PB1_CKPT}" ]] && { log "[P32 PREFLIGHT FAIL] missing PB1 ckpt: ${PB1_CKPT}"; preflight_fail=1; }
    fi
fi
if [[ ${preflight_fail} -ne 0 ]]; then
    log "[P32] FATAL preflight failures."
    exit 1
fi

log
log "================================================================"
log "R32 Phase 1 audit launch ${STAMP}"
log "DATASETS_ROOT=${DATASETS_ROOT}"
log "GPUS (CUDA_VISIBLE_DEVICES)=${GPUS}"
log "STAGE1P5_CKPT=${STAGE1P5_CKPT}"
log "PB1_CKPT=${PB1_CKPT}"
log "SELECTION_VAL=${SELECTION_VAL}"
log "BUCKETS=${BUCKETS_STR}"
log "SUB_DIR_ROOT=${SUB_DIR_ROOT}"
log "DIAG_DIR_ROOT=${DIAG_DIR_ROOT}"
log "AUDIT_OUT_DIR=${AUDIT_OUT_DIR}"
log "SKIP_SAMPLE=${SKIP_SAMPLE} SKIP_DOWNSTREAM=${SKIP_DOWNSTREAM} SKIP_AUDIT=${SKIP_AUDIT}"
log "================================================================"

# ─── Phase 1: SAMPLE (Stage-1.5 V0 → C41+S4 cache) ──────────────────
SAMPLE_DONE_MARKER="${SUB_DIR_ROOT}/val/.sample_done"

if [[ ${SKIP_SAMPLE} -eq 1 ]]; then
    log "[P32] --skip-sample: skipping sample phase"
elif [[ -f "${SAMPLE_DONE_MARKER}" && ${FORCE_RESAMPLE} -eq 0 ]]; then
    log "[P32] sample cache already present (${SAMPLE_DONE_MARKER}); skipping sample (use --force-resample to redo)"
else
    log
    log "================================================================"
    log "[$(date '+%F %T')] PHASE 1: SAMPLE Stage-1.5 → ${SUB_DIR_ROOT}"
    log "================================================================"

    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[P32 DRY-RUN] would invoke DS32 launcher with --skip-diag"
    else
        rm -rf "${SUB_DIR_ROOT}"

        set +e
        ROUND32_DS_STAGE1P5_CKPT="${STAGE1P5_CKPT}" \
        ROUND32_DS_PB1_CKPT="${PB1_CKPT}" \
        ROUND32_DS_BUCKETS="${BUCKETS_STR}" \
        ROUND32_DS_OUT_TAG="${OUT_TAG}" \
            bash scripts/stage_a_generator/run_round32_stage1p5_downstream_diag.sh \
                --skip-diag 2>&1 | tee -a "${SUMMARY_LOG}"
        rc=${PIPESTATUS[0]}
        set -e

        if [[ ${rc} -ne 0 ]]; then
            log "[P32] FATAL: sample phase exited rc=${rc}"
            exit "${rc}"
        fi
        # Record completion so a retry can skip.
        mkdir -p "${SUB_DIR_ROOT}/val"
        touch "${SAMPLE_DONE_MARKER}"
    fi
fi

# ─── Phase 2: DOWNSTREAM DIAG ──────────────────────────────────────
DIAG_DONE_MARKER="${DIAG_DIR_ROOT}/sustained_contact_val/sustained_contact_summary.md"

if [[ ${SKIP_DOWNSTREAM} -eq 1 ]]; then
    log "[P32] --skip-downstream: skipping downstream diag phase"
elif [[ -f "${DIAG_DONE_MARKER}" && ${FORCE_REDIAG} -eq 0 ]]; then
    log "[P32] downstream diag already archived (${DIAG_DONE_MARKER}); skipping (use --force-rediag)"
else
    log
    log "================================================================"
    log "[$(date '+%F %T')] PHASE 2: DOWNSTREAM DIAG -> ${DIAG_DIR_ROOT}"
    log "================================================================"

    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[P32 DRY-RUN] would invoke DS32 launcher with --skip-sample"
    else
        # The DS32 launcher writes to OUT_TAG-suffixed dirs; cache must be
        # already in place (phase 1 above).
        if [[ ! -d "${SUB_DIR_ROOT}/val" ]]; then
            log "[P32] FATAL: cache dir missing for downstream diag: ${SUB_DIR_ROOT}/val"
            exit 1
        fi

        # Clear any stale half-finished diag output before rerun.
        rm -rf "${DIAG_DIR_ROOT}"

        set +e
        ROUND32_DS_STAGE1P5_CKPT="${STAGE1P5_CKPT}" \
        ROUND32_DS_PB1_CKPT="${PB1_CKPT}" \
        ROUND32_DS_BUCKETS="${BUCKETS_STR}" \
        ROUND32_DS_OUT_TAG="${OUT_TAG}" \
            bash scripts/stage_a_generator/run_round32_stage1p5_downstream_diag.sh \
                --skip-sample 2>&1 | tee -a "${SUMMARY_LOG}"
        rc=${PIPESTATUS[0]}
        set -e

        if [[ ${rc} -ne 0 ]]; then
            log "[P32] FATAL: downstream-diag exited rc=${rc}"
            exit "${rc}"
        fi
    fi
fi

# ─── Phase 3: Distribution audit ───────────────────────────────────
AUDIT_DONE_MARKER="${AUDIT_OUT_DIR}/audit_report.md"

if [[ ${SKIP_AUDIT} -eq 1 ]]; then
    log "[P32] --skip-audit: skipping audit phase"
else
    log
    log "================================================================"
    log "[$(date '+%F %T')] PHASE 3: AUDIT -> ${AUDIT_OUT_DIR}"
    log "================================================================"

    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[P32 DRY-RUN] would invoke round32_phase1_dyn_audit.py"
        log "    upstream_dir = ${SUB_DIR_ROOT}/val"
        log "    out_dir      = ${AUDIT_OUT_DIR}"
    else
        if [[ ! -d "${SUB_DIR_ROOT}/val" ]]; then
            log "[P32] FATAL: cache dir missing for audit: ${SUB_DIR_ROOT}/val"
            exit 1
        fi

        mkdir -p "${AUDIT_OUT_DIR}"

        set +e
        "${PY}" -u scripts/stage_a_generator/round32_phase1_dyn_audit.py \
            --upstream-dir "${SUB_DIR_ROOT}/val" \
            --stage1p5-cfg "${STAGE1P5_CFG}" \
            --selection-json "${SELECTION_VAL}" \
            --out-dir "${AUDIT_OUT_DIR}" \
            2>&1 | tee -a "${SUMMARY_LOG}"
        rc=${PIPESTATUS[0]}
        set -e

        if [[ ${rc} -ne 0 ]]; then
            log "[P32] FATAL: audit script exited rc=${rc}"
            exit "${rc}"
        fi
    fi
fi

# ─── Phase 4: Pack everything ──────────────────────────────────────
if [[ ${DRY_RUN} -eq 0 ]]; then
    TARBALL="analyses/round32_phase1_audit_results_${STAMP}.tar.gz"
    log
    log "================================================================"
    log "[$(date '+%F %T')] PACK -> ${TARBALL}"
    log "================================================================"
    PACK_TARGETS=()
    [[ -d "${AUDIT_OUT_DIR}" ]] && PACK_TARGETS+=("${AUDIT_OUT_DIR}")
    [[ -d "${DIAG_DIR_ROOT}" ]] && PACK_TARGETS+=("${DIAG_DIR_ROOT}")
    [[ -d "${SUB_DIR_ROOT}" ]] && PACK_TARGETS+=("${SUB_DIR_ROOT}")
    [[ -d "${LOG_DIR}" ]] && PACK_TARGETS+=("${LOG_DIR}")
    if [[ ${#PACK_TARGETS[@]} -gt 0 ]]; then
        tar -czf "${TARBALL}" "${PACK_TARGETS[@]}"
        SIZE=$(du -h "${TARBALL}" | cut -f1)
        log "wrote ${TARBALL}  (${SIZE})"
    else
        log "[P32 PACK] nothing to pack"
    fi
fi

log
log "================================================================"
log "[$(date '+%F %T')] R32 Phase 1 audit COMPLETE"
log "================================================================"
log "audit_report : ${AUDIT_OUT_DIR}/audit_report.md"
log "downstream   : ${DIAG_DIR_ROOT}/*/...summary.md"
log "sample cache : ${SUB_DIR_ROOT}/val/"
log "summary log  : ${SUMMARY_LOG}"
if [[ ${DRY_RUN} -eq 0 ]]; then
    log "tarball      : ${TARBALL:-<none>}"
fi
