#!/usr/bin/env bash
# Full upstream pipeline: Stage-1 train → Stage-1.5 train → R31 B → R32 C → E2E D.
#
# Designed to run in a single tmux session (~6 h on dual-5080).
# Each phase is gated by the previous phase's artifact (final.pt). If
# any phase fails, the script exits and the next phase will not silently
# skip with stale data.
#
# Default behaviour: run all 5 phases sequentially.
#
# Skip flags (every phase has one; useful for resuming after a partial
# success or after testing one phase in isolation):
#   --skip-s1                skip Stage-1 train (assumes ckpt exists)
#   --skip-s1p5              skip Stage-1.5 train (assumes ckpt exists)
#   --skip-r31-diag          skip R31 B downstream diag
#   --skip-r32-diag          skip R32 C downstream diag
#   --skip-e2e-diag          skip end-to-end D diag
#   --only-train             run S1 + S1.5, no diag
#   --only-diag              skip both trains; run R31 + R32 + E2E
#
# Environment overrides flow through to the per-phase launchers, e.g.:
#   ROUND31_S1_NUM_PROCESSES=2      (Stage-1)
#   ROUND32_S1P5_NUM_PROCESSES=2    (Stage-1.5)
#   ROUND31_DS_SEED / ROUND32_DS_SEED / ROUND_E2E_SEED
#
# By default we assume 2× 5080 (dual-GPU train via accelerate). Set
# *_SINGLE_GPU=1 to force per-phase single-GPU.

set -euo pipefail
cd "$(dirname "$0")/../.."

SKIP_S1=0
SKIP_S1P5=0
SKIP_R31_DIAG=0
SKIP_R32_DIAG=0
SKIP_E2E_DIAG=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-s1)         SKIP_S1=1; shift ;;
        --skip-s1p5)       SKIP_S1P5=1; shift ;;
        --skip-r31-diag)   SKIP_R31_DIAG=1; shift ;;
        --skip-r32-diag)   SKIP_R32_DIAG=1; shift ;;
        --skip-e2e-diag)   SKIP_E2E_DIAG=1; shift ;;
        --only-train)
            SKIP_R31_DIAG=1; SKIP_R32_DIAG=1; SKIP_E2E_DIAG=1; shift ;;
        --only-diag)
            SKIP_S1=1; SKIP_S1P5=1; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help)         sed -n '1,40p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Default to 2 procs to match A1 / PB1's training schedule (bs=32 accum=1).
: "${ROUND31_S1_NUM_PROCESSES:=2}"
: "${ROUND32_S1P5_NUM_PROCESSES:=2}"
export ROUND31_S1_NUM_PROCESSES ROUND32_S1P5_NUM_PROCESSES

if [[ -z "${DATASETS_ROOT:-}" ]]; then
    echo "[PIPE] FATAL: DATASETS_ROOT must be exported before launch." >&2
    echo "       export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4" >&2
    exit 1
fi

STAGE1_CKPT="runs/training/stage1_traj_v0/final.pt"
STAGE1P5_CKPT="runs/training/stage1p5_interaction_v0/final.pt"
PB1_CKPT="runs/training/stageB_anchordiff_r29_pb_a1_adaln_s4/final.pt"
SELECTION_TRAIN="analyses/round27_tier0_train_indices_48_balanced.json"
SELECTION_VAL="analyses/round29_val_diag_indices_48_balanced.json"

OVERALL_LOG_DIR="runs/full_pipeline"
mkdir -p "${OVERALL_LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${OVERALL_LOG_DIR}/summary_${STAMP}.log"

log_summary() {
    echo "$@" | tee -a "${SUMMARY}"
}

phase_start() {
    local name="$1"
    log_summary
    log_summary "================================================================"
    log_summary "[$(date '+%F %T')] PHASE START: ${name}"
    log_summary "================================================================"
}

phase_done() {
    local name="$1"
    log_summary "[$(date '+%F %T')] PHASE DONE:  ${name}"
}

require_file() {
    local p="$1"; local what="$2"
    if [[ ! -e "${p}" ]]; then
        log_summary "[PIPE FATAL] ${what} missing: ${p}"
        exit 1
    fi
}

# ─── Preflight (catch obvious mistakes before any 1.5h train) ─────────
log_summary "===== Full upstream pipeline launch ${STAMP} ====="
log_summary "DATASETS_ROOT=${DATASETS_ROOT}"
log_summary "SKIP_S1=${SKIP_S1}  SKIP_S1P5=${SKIP_S1P5}"
log_summary "SKIP_R31_DIAG=${SKIP_R31_DIAG}  SKIP_R32_DIAG=${SKIP_R32_DIAG}  SKIP_E2E_DIAG=${SKIP_E2E_DIAG}"
log_summary "ROUND31_S1_NUM_PROCESSES=${ROUND31_S1_NUM_PROCESSES}  ROUND32_S1P5_NUM_PROCESSES=${ROUND32_S1P5_NUM_PROCESSES}"

# Datasets present.
if [[ ! -d "${DATASETS_ROOT}" ]]; then
    log_summary "[PIPE FATAL] DATASETS_ROOT not on disk: ${DATASETS_ROOT}"
    exit 1
fi

# Selection JSONs needed by all 3 diag phases.
if [[ ${SKIP_R31_DIAG} -eq 0 || ${SKIP_R32_DIAG} -eq 0 || ${SKIP_E2E_DIAG} -eq 0 ]]; then
    require_file "${SELECTION_TRAIN}" "train selection JSON"
    require_file "${SELECTION_VAL}"   "val selection JSON"
fi

# PB1 ckpt required for any diag phase (we don't retrain Stage-2).
if [[ ${SKIP_R31_DIAG} -eq 0 || ${SKIP_R32_DIAG} -eq 0 || ${SKIP_E2E_DIAG} -eq 0 ]]; then
    require_file "${PB1_CKPT}" "PB1 ckpt"
fi

# Generators present.
require_file "scripts/stage_a_generator/round31_make_stage1_configs.py" "S1 config generator"
require_file "scripts/stage_a_generator/round32_make_stage1p5_configs.py" "S1.5 config generator"

# Regenerate configs with server DATASETS_ROOT every run — cheap, idempotent.
log_summary "[PIPE] Regenerating Stage-1 + Stage-1.5 configs with server data root."
if [[ ${DRY_RUN} -eq 0 ]]; then
    python scripts/stage_a_generator/round31_make_stage1_configs.py \
        --data-root "${DATASETS_ROOT}" | tee -a "${SUMMARY}"
    python scripts/stage_a_generator/round32_make_stage1p5_configs.py \
        --data-root "${DATASETS_ROOT}" | tee -a "${SUMMARY}"
else
    log_summary "[DRY-RUN] would regenerate configs"
fi

# ─── PHASE 1: Stage-1 train ───────────────────────────────────────────
if [[ ${SKIP_S1} -eq 0 ]]; then
    phase_start "Stage-1 train (~1.5 h)"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log_summary "[DRY-RUN] bash scripts/stage_a_generator/run_round31_stage1_training.sh"
    else
        bash scripts/stage_a_generator/run_round31_stage1_training.sh
    fi
    phase_done  "Stage-1 train"
else
    log_summary "[PIPE] --skip-s1: skipping Stage-1 train"
fi
# Stage-1 ckpt must now exist for any downstream phase.
if [[ ${SKIP_S1P5} -eq 0 || ${SKIP_R31_DIAG} -eq 0 || ${SKIP_E2E_DIAG} -eq 0 ]]; then
    if [[ ${DRY_RUN} -eq 0 ]]; then
        require_file "${STAGE1_CKPT}" "Stage-1 ckpt (after Phase 1)"
    fi
fi

# ─── PHASE 2: Stage-1.5 train ─────────────────────────────────────────
if [[ ${SKIP_S1P5} -eq 0 ]]; then
    phase_start "Stage-1.5 train (~2 h)"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log_summary "[DRY-RUN] bash scripts/stage_a_generator/run_round32_stage1p5_training.sh"
    else
        bash scripts/stage_a_generator/run_round32_stage1p5_training.sh
    fi
    phase_done  "Stage-1.5 train"
else
    log_summary "[PIPE] --skip-s1p5: skipping Stage-1.5 train"
fi
if [[ ${SKIP_R32_DIAG} -eq 0 || ${SKIP_E2E_DIAG} -eq 0 ]]; then
    if [[ ${DRY_RUN} -eq 0 ]]; then
        require_file "${STAGE1P5_CKPT}" "Stage-1.5 ckpt (after Phase 2)"
    fi
fi

# ─── PHASE 3: R31 downstream diag (B) ─────────────────────────────────
if [[ ${SKIP_R31_DIAG} -eq 0 ]]; then
    phase_start "R31 Stage-1 downstream diag B (~50 min)"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log_summary "[DRY-RUN] bash scripts/stage_a_generator/run_round31_stage1_downstream_diag.sh"
    else
        bash scripts/stage_a_generator/run_round31_stage1_downstream_diag.sh
    fi
    phase_done  "R31 B"
else
    log_summary "[PIPE] --skip-r31-diag: skipping R31 B"
fi

# ─── PHASE 4: R32 downstream diag (C) ─────────────────────────────────
if [[ ${SKIP_R32_DIAG} -eq 0 ]]; then
    phase_start "R32 Stage-1.5 downstream diag C (~70 min)"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log_summary "[DRY-RUN] bash scripts/stage_a_generator/run_round32_stage1p5_downstream_diag.sh"
    else
        bash scripts/stage_a_generator/run_round32_stage1p5_downstream_diag.sh
    fi
    phase_done  "R32 C"
else
    log_summary "[PIPE] --skip-r32-diag: skipping R32 C"
fi

# ─── PHASE 5: End-to-end diag (D) — reuse Stage-1 cache from Phase 3 ──
if [[ ${SKIP_E2E_DIAG} -eq 0 ]]; then
    phase_start "End-to-end D (~70 min, reuses R31's Stage-1 cache)"
    REUSE_FLAG=""
    if [[ -d "analyses/round31_stage1_substitute_conds/val" ]]; then
        REUSE_FLAG="--reuse-stage1-cache"
        log_summary "[PIPE] Reusing R31's Stage-1 cache (saves ~16 min sampling)"
    else
        log_summary "[PIPE] No R31 Stage-1 cache found; end-to-end will sample from scratch"
    fi
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log_summary "[DRY-RUN] bash scripts/stage_a_generator/run_round31_32_end_to_end_diag.sh ${REUSE_FLAG}"
    else
        bash scripts/stage_a_generator/run_round31_32_end_to_end_diag.sh ${REUSE_FLAG}
    fi
    phase_done  "End-to-end D"
else
    log_summary "[PIPE] --skip-e2e-diag: skipping end-to-end D"
fi

# ─── Final summary ────────────────────────────────────────────────────
log_summary
log_summary "================================================================"
log_summary "[$(date '+%F %T')] FULL PIPELINE COMPLETE"
log_summary "================================================================"
log_summary "Tarballs produced (any that are missing failed silently):"
for t in analyses/round31_stage1_results_*.tar.gz \
         analyses/round32_stage1p5_results_*.tar.gz \
         analyses/round31_stage1_downstream_results_*.tar.gz \
         analyses/round32_stage1p5_downstream_results_*.tar.gz \
         analyses/round31_32_end_to_end_results_*.tar.gz; do
    if compgen -G "${t}" > /dev/null; then
        for f in ${t}; do
            SIZE=$(du -h "${f}" | cut -f1)
            log_summary "  ${f}  (${SIZE})"
        done
    fi
done
log_summary
log_summary "Summary log:  ${SUMMARY}"
log_summary "scp back from local:"
log_summary "  scp <server>:$(pwd)/analyses/round31_stage1_results_*.tar.gz                 ./analyses/"
log_summary "  scp <server>:$(pwd)/analyses/round32_stage1p5_results_*.tar.gz               ./analyses/"
log_summary "  scp <server>:$(pwd)/analyses/round31_stage1_downstream_results_*.tar.gz      ./analyses/"
log_summary "  scp <server>:$(pwd)/analyses/round32_stage1p5_downstream_results_*.tar.gz    ./analyses/"
log_summary "  scp <server>:$(pwd)/analyses/round31_32_end_to_end_results_*.tar.gz          ./analyses/"
