#!/usr/bin/env bash
# Round-31 Stage-1 downstream-coupling diagnostic launcher (B in design doc).
#
# Pipes Stage-1's sampled stage1_coarse into frozen PB1 (Stage-2 ckpt),
# then runs Stage-2's 4 standard diag kinds (sustained_contact / gait /
# body_action / g1_soft_stance) × {train, val} on the same 48-clip
# selections as the original PB1 diag. Comparing these metrics to PB1's
# oracle-cond metrics (already in tree under
# analyses/round29_r29_pb_a1_adaln_s4_diag_*_val/) tells us how much
# Stage-1's imperfect output degrades downstream Stage-2 quality.
#
# Phases:
#   1) Sample Stage-1 outputs on the train + val selections → cache
#      under analyses/round31_stage1_substitute_conds/{train,val}/
#   2) Run the 4 Stage-2 diag scripts with
#      --substitute-conds-dir pointing at that cache.
#   3) Pack everything into a tarball.
#
# Usage:
#   bash scripts/stage_a_generator/run_round31_stage1_downstream_diag.sh
#   bash scripts/stage_a_generator/run_round31_stage1_downstream_diag.sh --dry-run
#   bash scripts/stage_a_generator/run_round31_stage1_downstream_diag.sh --only val
#   bash scripts/stage_a_generator/run_round31_stage1_downstream_diag.sh --skip-sample
#
# Env overrides:
#   ROUND31_DS_NUM_PROCESSES=N           (unused — sampling is single-GPU per bucket)
#   ROUND31_DS_ALLOW_PARTIAL=1           don't FATAL on missing diag kinds
#   ROUND31_DS_SEED=42                   sampling seed
#   ROUND31_DS_CFG_SCALE=1.0             default CFG (overridden by next two if set)
#   ROUND31_DS_STAGE1_CFG_SCALE=...      CFG at Stage-1 sample time (Stage-1's knob)
#   ROUND31_DS_PB1_CFG_SCALE=...         CFG at PB1 diag time (PB1's knob, keep 1.0
#                                        to compare apples-to-apples vs PB1 oracle)
#   ROUND31_DS_SAMPLER=ddim_eta0         sampler at Stage-1 sample time only
#                                        (PB1 diag side uses model.sample() default,
#                                        no per-diag sampler knob.)
#   ROUND31_DS_STAGE1_SAMPLER=...        explicit alias for Stage-1 sampler
#   ROUND31_DS_STAGE1_CKPT=...           override default Stage-1 ckpt path
#   ROUND31_DS_PB1_CKPT=...              override default PB1 ckpt path
#   ROUND31_DS_BUCKETS="train val"       which buckets to run
#   ROUND31_DS_OUT_TAG=""                tag appended to output dirs (allows multiple
#                                        sweep runs without clobbering each other)
#
# Prerequisites:
#   - Stage-1 trained: runs/training/stage1_traj_v0/final.pt
#   - Stage-2 PB1 trained: runs/training/stageB_anchordiff_r29_pb_a1_adaln_s4/final.pt
#   - Selection JSONs: analyses/round27_tier0_train_indices_48_balanced.json
#                      analyses/round29_val_diag_indices_48_balanced.json

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
SKIP_SAMPLE=0
SKIP_DIAG=0
ALLOW_PARTIAL="${ROUND31_DS_ALLOW_PARTIAL:-0}"
SEED="${ROUND31_DS_SEED:-42}"
CFG_SCALE="${ROUND31_DS_CFG_SCALE:-1.0}"
STAGE1_CFG_SCALE="${ROUND31_DS_STAGE1_CFG_SCALE:-${CFG_SCALE}}"
PB1_CFG_SCALE="${ROUND31_DS_PB1_CFG_SCALE:-${CFG_SCALE}}"
SAMPLER="${ROUND31_DS_SAMPLER:-ddim_eta0}"
STAGE1_SAMPLER="${ROUND31_DS_STAGE1_SAMPLER:-${SAMPLER}}"
STAGE1_CFG="${ROUND31_DS_STAGE1_CFG:-configs/training/stage1_traj_v0.yaml}"
STAGE1_CKPT="${ROUND31_DS_STAGE1_CKPT:-runs/training/stage1_traj_v0/final.pt}"
PB1_VARIANT="r29_pb_a1_adaln_s4"
PB1_CFG="configs/training/anchordiff_${PB1_VARIANT}.yaml"
PB1_CKPT="${ROUND31_DS_PB1_CKPT:-runs/training/stageB_anchordiff_${PB1_VARIANT}/final.pt}"
BUCKETS_STR="${ROUND31_DS_BUCKETS:-train val}"
OUT_TAG="${ROUND31_DS_OUT_TAG:-}"
SELECTION_TRAIN="analyses/round27_tier0_train_indices_48_balanced.json"
SELECTION_VAL="analyses/round29_val_diag_indices_48_balanced.json"
SUB_DIR_ROOT="analyses/round31_stage1_substitute_conds${OUT_TAG}"
DIAG_DIR_ROOT="analyses/round31_stage1_downstream_diag${OUT_TAG}"
LOG_DIR="runs/round31_stage1_downstream${OUT_TAG}"
mkdir -p "${LOG_DIR}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)           ONLY="$2"; shift 2 ;;       # "train", "val", or empty
        --dry-run)        DRY_RUN=1; shift ;;
        --skip-sample)    SKIP_SAMPLE=1; shift ;;
        --skip-diag)      SKIP_DIAG=1; shift ;;
        -h|--help)        sed -n '1,40p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python";
    elif command -v python3 >/dev/null 2>&1; then PY="python3";
    else echo "[DS31] FATAL: neither python nor python3 was found" >&2; exit 127; fi
fi

if [[ -n "${ONLY}" ]]; then
    BUCKETS_STR="${ONLY}"
fi
# shellcheck disable=SC2206
BUCKETS=(${BUCKETS_STR})

echo "[DS31] STAGE1_CKPT=${STAGE1_CKPT}"
echo "[DS31] PB1_CKPT=${PB1_CKPT}"
echo "[DS31] BUCKETS=${BUCKETS[*]}  SEED=${SEED}"
echo "[DS31] Stage-1 sample: cfg=${STAGE1_CFG_SCALE} sampler=${STAGE1_SAMPLER}"
echo "[DS31] PB1     diag  : cfg=${PB1_CFG_SCALE} sampler=model.sample-default"
echo "[DS31] OUT_TAG=${OUT_TAG}  SUB_DIR=${SUB_DIR_ROOT}  DIAG_DIR=${DIAG_DIR_ROOT}"

# ─── Preflight ────────────────────────────────────────────────────────
# Under --dry-run we tolerate missing ckpts/selections so the script can be
# invoked from a laptop to sanity-check env passthrough.
preflight_fail=0
for p in "${STAGE1_CFG}" "${PB1_CFG}"; do
    [[ ! -e "${p}" ]] && { echo "[DS31 PREFLIGHT FAIL] missing config: ${p}"; preflight_fail=1; }
done
if [[ ${DRY_RUN} -eq 0 ]]; then
    for p in "${STAGE1_CKPT}" "${PB1_CKPT}"; do
        [[ ! -e "${p}" ]] && { echo "[DS31 PREFLIGHT FAIL] missing ckpt: ${p}"; preflight_fail=1; }
    done
    for b in "${BUCKETS[@]}"; do
        case "${b}" in
            train) sel="${SELECTION_TRAIN}" ;;
            val)   sel="${SELECTION_VAL}"   ;;
            *) echo "[DS31 PREFLIGHT FAIL] unknown bucket: ${b}"; preflight_fail=1; continue ;;
        esac
        [[ ! -e "${sel}" ]] && { echo "[DS31 PREFLIGHT FAIL] missing selection: ${sel}"; preflight_fail=1; }
    done
fi
for s in scripts/stage_b_generator/round26_sustained_contact_diag.py \
         scripts/stage_b_generator/round26_gait_diag.py \
         scripts/stage_b_generator/round28_body_action_diag.py \
         scripts/stage_b_generator/round29_g1_soft_stance_diag.py \
         scripts/stage_a_generator/sample_substitute_conds_cli.py; do
    [[ ! -e "${s}" ]] && { echo "[DS31 PREFLIGHT FAIL] missing script: ${s}"; preflight_fail=1; }
done
if [[ ${preflight_fail} -ne 0 ]]; then
    echo "[DS31] FATAL preflight failures."
    exit 1
fi

# ─── Phase 1: Sample Stage-1 outputs on each bucket ───────────────────
for b in "${BUCKETS[@]}"; do
    case "${b}" in
        train) sel="${SELECTION_TRAIN}" ;;
        val)   sel="${SELECTION_VAL}"   ;;
    esac
    SUB_DIR="${SUB_DIR_ROOT}/${b}"
    LOG="${LOG_DIR}/sample_stage1_${b}.log"

    if [[ ${SKIP_SAMPLE} -eq 1 ]]; then
        echo "[DS31] --skip-sample: skipping sampling for ${b}"
        continue
    fi

    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE 1: SAMPLE Stage-1 → ${SUB_DIR}"
    echo "================================================================"
    SAMPLE_CMD=("${PY}" -u scripts/stage_a_generator/sample_substitute_conds_cli.py
        --stage stage1
        --config "${STAGE1_CFG}"
        --ckpt "${STAGE1_CKPT}"
        --selection-json "${sel}"
        --bucket "${b}"
        --out-dir "${SUB_DIR}"
        --seed "${SEED}"
        --cfg-scale "${STAGE1_CFG_SCALE}"
        --sampler "${STAGE1_SAMPLER}")
    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "[DS31 DRY-RUN]"
        echo "    \$ ${SAMPLE_CMD[*]}"
    else
        "${SAMPLE_CMD[@]}" 2>&1 | tee "${LOG}"
    fi
done

# ─── Phase 2: Run the 4 Stage-2 diags with substitute_conds_dir ───────
run_diag() {
    local KIND="$1"; local BUCKET="$2"; local SUB_DIR="$3"
    case "${BUCKET}" in
        train) sel="${SELECTION_TRAIN}" ;;
        val)   sel="${SELECTION_VAL}"   ;;
    esac
    case "${KIND}" in
        sustained_contact) SCRIPT="scripts/stage_b_generator/round26_sustained_contact_diag.py" ;;
        gait)              SCRIPT="scripts/stage_b_generator/round26_gait_diag.py" ;;
        body_action)       SCRIPT="scripts/stage_b_generator/round28_body_action_diag.py" ;;
        g1_soft_stance)    SCRIPT="scripts/stage_b_generator/round29_g1_soft_stance_diag.py" ;;
        *) echo "[DS31 DIAG] unknown kind ${KIND}"; return 1 ;;
    esac
    OUT_DIR="${DIAG_DIR_ROOT}/${KIND}_${BUCKET}"
    LOG="${LOG_DIR}/diag_${KIND}_${BUCKET}.log"
    mkdir -p "${OUT_DIR}"

    CMD=("${PY}" -u "${SCRIPT}"
        --config "${PB1_CFG}" --ckpt "${PB1_CKPT}"
        --selection-json "${sel}"
        --output-dir "${OUT_DIR}"
        --bucket "${BUCKET}"
        --substitute-conds-dir "${SUB_DIR}"
        --cfg-scale "${PB1_CFG_SCALE}" --seed "${SEED}")
    echo "[DS31 DIAG START] ${KIND} ${BUCKET}"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "    \$ ${CMD[*]}"
        return 0
    fi
    if ! "${CMD[@]}" 2>&1 | tee "${LOG}"; then
        if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
            echo "[DS31] WARN: ${KIND} ${BUCKET} failed; continuing (ALLOW_PARTIAL=1)"
        else
            echo "[DS31] FATAL: ${KIND} ${BUCKET} failed."
            return 1
        fi
    fi
}

if [[ ${SKIP_DIAG} -eq 1 ]]; then
    echo "[DS31] --skip-diag: skipping Phase 2"
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE 2: DIAG (4 kinds × ${#BUCKETS[@]} buckets)"
    echo "================================================================"
    for b in "${BUCKETS[@]}"; do
        SUB_DIR="${SUB_DIR_ROOT}/${b}"
        for KIND in sustained_contact gait body_action g1_soft_stance; do
            run_diag "${KIND}" "${b}" "${SUB_DIR}"
        done
    done
fi

# ─── Phase 3: Pack ───────────────────────────────────────────────────
if [[ ${DRY_RUN} -eq 0 ]]; then
    STAMP=$(date +%Y%m%d_%H%M%S)
    TARBALL="analyses/round31_stage1_downstream_results${OUT_TAG}_${STAMP}.tar.gz"
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PACK -> ${TARBALL}"
    echo "================================================================"
    PACK_TARGETS=()
    for b in "${BUCKETS[@]}"; do
        for KIND in sustained_contact gait body_action g1_soft_stance; do
            D="${DIAG_DIR_ROOT}/${KIND}_${b}"
            [[ -d "${D}" ]] && PACK_TARGETS+=("${D}")
        done
    done
    [[ -d "${SUB_DIR_ROOT}" ]] && PACK_TARGETS+=("${SUB_DIR_ROOT}")
    [[ -d "${LOG_DIR}" ]] && PACK_TARGETS+=("${LOG_DIR}")
    if [[ ${#PACK_TARGETS[@]} -gt 0 ]]; then
        tar -czf "${TARBALL}" "${PACK_TARGETS[@]}"
        SIZE=$(du -h "${TARBALL}" | cut -f1)
        echo "wrote ${TARBALL}  (${SIZE})"
    else
        echo "[DS31 PACK] nothing to pack"
    fi
fi

echo
echo "================================================================"
echo "Round-31 Stage-1 downstream diag complete."
echo "================================================================"
