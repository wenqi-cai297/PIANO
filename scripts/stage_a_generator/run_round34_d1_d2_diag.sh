#!/usr/bin/env bash
# Round-34 D1 (spectral-swap) + D2 (residual-sensitivity) — no-train causal diag.
#
# Workflow (brief §5 + §6):
#   Phase 0 — dump oracle GT (C41, S4) for the 48-clip val selection.
#   Phase 1 — build D1 7 variant substitute_conds dirs (mix gt + pred per rule).
#   Phase 2 — build D2 30 variant substitute_conds dirs (alpha-residual sweep).
#   Phase 3 — for each variant dir, run the 4 frozen-PB1 diagnostics
#             (sustained_contact / gait / body_action / g1_soft_stance) via
#             ``round32_stage1p5_downstream_diag.sh --skip-sample`` (we already
#             have the cond cache; no Stage-1.5 sampling needed).
#   Phase 4 — pack each variant's diag dir + write the summary tarball.
#
# No Stage-1.5 / PB1 training. No gradient updates. Read-only PB1 inference.

set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── Knobs ─────────────────────────────────────────────────────────────────
PB1_CFG="${PB1_CFG:-configs/training/anchordiff_r29_pb_a1_adaln_s4.yaml}"
PB1_CKPT="${PB1_CKPT:-runs/training/stageB_anchordiff_r29_pb_a1_adaln_s4/final.pt}"
SELECTION_JSON="${SELECTION_JSON:-analyses/round29_val_diag_indices_48_balanced.json}"
BUCKET="${BUCKET:-val}"
PRED_DIR="${PRED_DIR:-analyses/round32_stage1p5_substitute_conds_r33_stage1p5_r33_v1_xattn}"
WORK_ROOT="${WORK_ROOT:-analyses/2026-05-31_stage1p5_wrist_external_review_work}"
GT_DUMP_DIR="${WORK_ROOT}/oracle_dump"
D1_VARIANT_ROOT="${WORK_ROOT}/d1_variants"
D2_VARIANT_ROOT="${WORK_ROOT}/d2_variants"
D1_DIAG_ROOT="${WORK_ROOT}/d1_diag"
D2_DIAG_ROOT="${WORK_ROOT}/d2_diag"
LOG_DIR="${WORK_ROOT}/logs"
mkdir -p "${LOG_DIR}"

DO_PHASES="${DO_PHASES:-0,1,2,3,4}"    # comma-list; allows skipping
SEED="${SEED:-42}"
CFG_SCALE="${CFG_SCALE:-1.0}"
SAMPLER="${SAMPLER:-ddim_eta0}"        # unused (we --skip-sample) but DS32 wants it

D1_VARIANTS="${D1_VARIANTS:-v0,v1,v2,v3,v4,v5,v6,v7}"
D2_GROUPS="${D2_GROUPS:-g0,g1,g2,g3,g4,g5}"
D2_ALPHAS="${D2_ALPHAS:-0.10,0.25,0.50,0.75,1.00}"

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python";
    elif command -v python3 >/dev/null 2>&1; then PY="python3";
    else echo "[D1D2] FATAL: no python found" >&2; exit 127; fi
fi
phase_active() {
    local p="$1"
    [[ ",${DO_PHASES}," == *",${p},"* ]]
}

# ─── Preflight ─────────────────────────────────────────────────────────────
echo "[D1D2] preflight"
for f in "${PB1_CFG}" "${SELECTION_JSON}"; do
    [[ ! -e "$f" ]] && { echo "[D1D2] FATAL: missing $f"; exit 1; }
done
if phase_active 3; then
    [[ ! -e "${PB1_CKPT}" ]] && { echo "[D1D2] FATAL: missing PB1 ckpt ${PB1_CKPT}"; exit 1; }
fi
for s in \
    scripts/stage_a_generator/dump_gt_c41_s4_oracle.py \
    scripts/stage_a_generator/build_d1_spectral_swap_variants.py \
    scripts/stage_a_generator/build_d2_residual_sensitivity_variants.py \
    scripts/stage_a_generator/run_round32_stage1p5_downstream_diag.sh \
    scripts/stage_b_generator/round26_sustained_contact_diag.py \
    scripts/stage_b_generator/round26_gait_diag.py \
    scripts/stage_b_generator/round28_body_action_diag.py \
    scripts/stage_b_generator/round29_g1_soft_stance_diag.py; do
    [[ ! -e "$s" ]] && { echo "[D1D2] FATAL: missing $s"; exit 1; }
done

# ─── Phase 0: GT dump ──────────────────────────────────────────────────────
if phase_active 0; then
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE 0 — dump oracle GT C41/S4 → ${GT_DUMP_DIR}"
    echo "================================================================"
    LOG="${LOG_DIR}/phase0_gt_dump.log"
    "${PY}" -u scripts/stage_a_generator/dump_gt_c41_s4_oracle.py \
        --cfg "${PB1_CFG}" \
        --selection-json "${SELECTION_JSON}" \
        --bucket "${BUCKET}" \
        --out-dir "${GT_DUMP_DIR}" 2>&1 | tee "${LOG}"
fi

# ─── Phase 1: D1 variant build ─────────────────────────────────────────────
if phase_active 1; then
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE 1 — build D1 variants → ${D1_VARIANT_ROOT}"
    echo "================================================================"
    LOG="${LOG_DIR}/phase1_d1_build.log"
    "${PY}" -u scripts/stage_a_generator/build_d1_spectral_swap_variants.py \
        --gt-dir "${GT_DUMP_DIR}" \
        --pred-dir "${PRED_DIR}" \
        --bucket "${BUCKET}" \
        --variants "${D1_VARIANTS}" \
        --out-root "${D1_VARIANT_ROOT}" 2>&1 | tee "${LOG}"
fi

# ─── Phase 2: D2 variant build ─────────────────────────────────────────────
if phase_active 2; then
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE 2 — build D2 variants → ${D2_VARIANT_ROOT}"
    echo "================================================================"
    LOG="${LOG_DIR}/phase2_d2_build.log"
    "${PY}" -u scripts/stage_a_generator/build_d2_residual_sensitivity_variants.py \
        --gt-dir "${GT_DUMP_DIR}" \
        --pred-dir "${PRED_DIR}" \
        --bucket "${BUCKET}" \
        --groups "${D2_GROUPS}" \
        --alphas "${D2_ALPHAS}" \
        --out-root "${D2_VARIANT_ROOT}" 2>&1 | tee "${LOG}"
fi

# ─── Diag runner: invoke one of the 4 PB1 diagnostics on a SUB dir ──────
# We call the diag scripts directly with --substitute-conds-dir (no need to
# go through the DS32 launcher, since we --skip-sample anyway).
run_pb1_diag() {
    local KIND="$1"; local SUB_DIR="$2"; local OUT_DIR="$3"; local LOG="$4"
    local SCRIPT=""
    case "${KIND}" in
        sustained_contact) SCRIPT="scripts/stage_b_generator/round26_sustained_contact_diag.py" ;;
        gait)              SCRIPT="scripts/stage_b_generator/round26_gait_diag.py" ;;
        body_action)       SCRIPT="scripts/stage_b_generator/round28_body_action_diag.py" ;;
        g1_soft_stance)    SCRIPT="scripts/stage_b_generator/round29_g1_soft_stance_diag.py" ;;
        *) echo "[D1D2] unknown kind ${KIND}"; return 1 ;;
    esac
    mkdir -p "${OUT_DIR}"
    "${PY}" -u "${SCRIPT}" \
        --config "${PB1_CFG}" --ckpt "${PB1_CKPT}" \
        --selection-json "${SELECTION_JSON}" \
        --output-dir "${OUT_DIR}" \
        --bucket "${BUCKET}" \
        --substitute-conds-dir "${SUB_DIR}" \
        --cfg-scale "${CFG_SCALE}" --seed "${SEED}" 2>&1 | tee "${LOG}"
}

# Iterate one variant dir through all 4 diags.
run_all_diags_for_variant() {
    local VAR_TAG="$1"; local SUB_DIR_ROOT="$2"; local DIAG_DIR_ROOT="$3"
    local SUB_DIR="${SUB_DIR_ROOT}/${BUCKET}"
    if [[ ! -d "${SUB_DIR}" ]]; then
        echo "[D1D2] skip ${VAR_TAG} — missing ${SUB_DIR}"
        return 0
    fi
    echo
    echo "──── ${VAR_TAG} ────"
    for KIND in sustained_contact gait body_action g1_soft_stance; do
        local OUT_DIR="${DIAG_DIR_ROOT}/${VAR_TAG}/${KIND}_${BUCKET}"
        local LOG="${LOG_DIR}/diag_${VAR_TAG}_${KIND}_${BUCKET}.log"
        # Skip if summary already exists (idempotency).
        if [[ -e "${OUT_DIR}/${KIND}_summary.md" ]]; then
            echo "[D1D2] ${VAR_TAG} ${KIND}: existing summary; skipping"
            continue
        fi
        run_pb1_diag "${KIND}" "${SUB_DIR_ROOT}/${BUCKET}" "${OUT_DIR}" "${LOG}"
    done
}

# ─── Phase 3: D1 + D2 PB1 diag runs ────────────────────────────────────────
if phase_active 3; then
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE 3 — PB1 diag over D1 variants"
    echo "================================================================"
    IFS=',' read -r -a D1_VARS <<< "${D1_VARIANTS}"
    for v in "${D1_VARS[@]}"; do
        run_all_diags_for_variant "${v}" "${D1_VARIANT_ROOT}/${v}" "${D1_DIAG_ROOT}"
    done

    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE 3 — PB1 diag over D2 variants"
    echo "================================================================"
    IFS=',' read -r -a D2_GRPS <<< "${D2_GROUPS}"
    IFS=',' read -r -a D2_ALPS <<< "${D2_ALPHAS}"
    for g in "${D2_GRPS[@]}"; do
        for a in "${D2_ALPS[@]}"; do
            ALPHA_TAG="alpha$(printf '%03d' "$(printf '%.0f' "$(echo "${a} * 100" | bc -l)")")"
            TAG="${g}_${ALPHA_TAG}"
            run_all_diags_for_variant "${TAG}" "${D2_VARIANT_ROOT}/${TAG}" "${D2_DIAG_ROOT}"
        done
    done
fi

# ─── Phase 4: pack ─────────────────────────────────────────────────────────
if phase_active 4; then
    STAMP=$(date +%Y%m%d_%H%M%S)
    D1_TAR="analyses/round34_d1_diag_results_${STAMP}.tar.gz"
    D2_TAR="analyses/round34_d2_diag_results_${STAMP}.tar.gz"
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE 4 — pack"
    echo "================================================================"
    if [[ -d "${D1_DIAG_ROOT}" ]]; then
        tar -czf "${D1_TAR}" "${D1_DIAG_ROOT}" "${LOG_DIR}" \
            && echo "  wrote ${D1_TAR}"
    fi
    if [[ -d "${D2_DIAG_ROOT}" ]]; then
        tar -czf "${D2_TAR}" "${D2_DIAG_ROOT}" "${LOG_DIR}" \
            && echo "  wrote ${D2_TAR}"
    fi
fi

echo
echo "================================================================"
echo "Round-34 D1+D2 diag complete: $(date '+%F %T')"
echo "================================================================"
