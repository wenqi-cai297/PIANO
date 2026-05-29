#!/usr/bin/env bash
# End-to-end downstream-coupling diagnostic launcher (D in design doc).
#
# Pipes:
#   Stage-1 sample → Stage-1.5 sample (using Stage-1 output as cond) →
#   frozen PB1 (using both as cond).
# Compares to PB1 with all-oracle conds (baseline A).
#
# Phases:
#   1a) Sample Stage-1 outputs on each bucket (= R31 Phase 1).
#       If R31 already ran and the cache exists, --reuse-stage1-cache
#       skips this phase.
#   1b) Sample Stage-1.5 outputs USING Stage-1's cache as the
#       stage1_coarse cond (--upstream-dir).
#   2)  Run 4 Stage-2 diags with --substitute-conds-dir pointing at the
#       Stage-1.5-on-Stage-1 cache. The Stage-2 diag's _build_cond reads
#       BOTH the stage1_coarse and (C41, S4) substitutes for each clip,
#       but they're produced from the same cache layer in (1b) — Stage-1.5
#       writes stage2_coarse_extra + stage2_support, and we ALSO need
#       Stage-1's stage1_coarse for the same clip. So the substitute_dir
#       passed to the diag is a MERGED dir.
#   3)  Pack.

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
SKIP_SAMPLE_STAGE1=0
SKIP_SAMPLE_STAGE1P5=0
SKIP_DIAG=0
ALLOW_PARTIAL="${ROUND_E2E_ALLOW_PARTIAL:-0}"
SEED="${ROUND_E2E_SEED:-42}"
CFG_SCALE="${ROUND_E2E_CFG_SCALE:-1.0}"
SAMPLER="${ROUND_E2E_SAMPLER:-ddim_eta0}"
STAGE1_CFG="${ROUND_E2E_STAGE1_CFG:-configs/training/stage1_traj_v0.yaml}"
STAGE1_CKPT="${ROUND_E2E_STAGE1_CKPT:-runs/training/stage1_traj_v0/final.pt}"
STAGE1P5_CFG="configs/training/stage1p5_interaction_v0.yaml"
STAGE1P5_CKPT="${ROUND_E2E_STAGE1P5_CKPT:-runs/training/stage1p5_interaction_v0/final.pt}"
PB1_VARIANT="r29_pb_a1_adaln_s4"
PB1_CFG="configs/training/anchordiff_${PB1_VARIANT}.yaml"
PB1_CKPT="${ROUND_E2E_PB1_CKPT:-runs/training/stageB_anchordiff_${PB1_VARIANT}/final.pt}"
BUCKETS_STR="${ROUND_E2E_BUCKETS:-train val}"
SELECTION_TRAIN="analyses/round27_tier0_train_indices_48_balanced.json"
SELECTION_VAL="analyses/round29_val_diag_indices_48_balanced.json"
# Reuse R31's Stage-1 cache if it exists; the merged dir holds Stage-1.5
# output + a SYMLINK to Stage-1 output (via copy).
S1_CACHE_ROOT="${ROUND_E2E_S1_CACHE_ROOT:-analyses/round31_stage1_substitute_conds}"
E2E_CACHE_ROOT="analyses/round31_32_end_to_end_substitute_conds"
LOG_DIR="runs/round31_32_end_to_end"
mkdir -p "${LOG_DIR}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)                ONLY="$2"; shift 2 ;;
        --dry-run)             DRY_RUN=1; shift ;;
        --skip-sample-stage1)  SKIP_SAMPLE_STAGE1=1; shift ;;
        --skip-sample-stage1p5) SKIP_SAMPLE_STAGE1P5=1; shift ;;
        --skip-diag)           SKIP_DIAG=1; shift ;;
        --reuse-stage1-cache)  SKIP_SAMPLE_STAGE1=1; shift ;;
        -h|--help)             sed -n '1,40p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python";
    elif command -v python3 >/dev/null 2>&1; then PY="python3";
    else echo "[E2E] FATAL: neither python nor python3 was found" >&2; exit 127; fi
fi
if [[ -n "${ONLY}" ]]; then BUCKETS_STR="${ONLY}"; fi
# shellcheck disable=SC2206
BUCKETS=(${BUCKETS_STR})

echo "[E2E] STAGE1_CKPT=${STAGE1_CKPT}"
echo "[E2E] STAGE1P5_CKPT=${STAGE1P5_CKPT}"
echo "[E2E] PB1_CKPT=${PB1_CKPT}"
echo "[E2E] BUCKETS=${BUCKETS[*]}  SAMPLER=${SAMPLER}  SEED=${SEED}  CFG_SCALE=${CFG_SCALE}"

preflight_fail=0
for p in "${STAGE1_CFG}" "${STAGE1P5_CFG}" "${PB1_CFG}"; do
    [[ ! -e "${p}" ]] && { echo "[E2E PREFLIGHT FAIL] missing config: ${p}"; preflight_fail=1; }
done
for p in "${STAGE1_CKPT}" "${STAGE1P5_CKPT}" "${PB1_CKPT}"; do
    [[ ! -e "${p}" ]] && { echo "[E2E PREFLIGHT FAIL] missing ckpt: ${p}"; preflight_fail=1; }
done
for b in "${BUCKETS[@]}"; do
    case "${b}" in
        train) sel="${SELECTION_TRAIN}" ;;
        val)   sel="${SELECTION_VAL}"   ;;
        *) echo "[E2E PREFLIGHT FAIL] unknown bucket: ${b}"; preflight_fail=1; continue ;;
    esac
    [[ ! -e "${sel}" ]] && { echo "[E2E PREFLIGHT FAIL] missing selection: ${sel}"; preflight_fail=1; }
done
if [[ ${preflight_fail} -ne 0 ]]; then echo "[E2E] FATAL preflight failures."; exit 1; fi

# Phase 1a: Sample Stage-1 (if cache missing or --reuse-stage1-cache off).
for b in "${BUCKETS[@]}"; do
    case "${b}" in train) sel="${SELECTION_TRAIN}" ;; val) sel="${SELECTION_VAL}" ;; esac
    S1_CACHE="${S1_CACHE_ROOT}/${b}"
    LOG="${LOG_DIR}/sample_stage1_${b}.log"
    if [[ ${SKIP_SAMPLE_STAGE1} -eq 1 ]]; then
        echo "[E2E] --skip-sample-stage1 / --reuse-stage1-cache: using ${S1_CACHE}"
        if [[ ! -d "${S1_CACHE}" ]]; then
            echo "[E2E PREFLIGHT FAIL] Stage-1 cache missing: ${S1_CACHE}"; exit 1
        fi
        continue
    fi
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE 1a: SAMPLE Stage-1 → ${S1_CACHE}"
    echo "================================================================"
    SAMPLE_CMD=("${PY}" -u scripts/stage_a_generator/sample_substitute_conds_cli.py
        --stage stage1 --config "${STAGE1_CFG}" --ckpt "${STAGE1_CKPT}"
        --selection-json "${sel}" --bucket "${b}" --out-dir "${S1_CACHE}"
        --seed "${SEED}" --cfg-scale "${CFG_SCALE}" --sampler "${SAMPLER}")
    if [[ ${DRY_RUN} -eq 1 ]]; then echo "[E2E DRY-RUN]"; echo "    \$ ${SAMPLE_CMD[*]}";
    else "${SAMPLE_CMD[@]}" 2>&1 | tee "${LOG}"; fi
done

# Phase 1b: Sample Stage-1.5 USING Stage-1 cache as cond.
for b in "${BUCKETS[@]}"; do
    case "${b}" in train) sel="${SELECTION_TRAIN}" ;; val) sel="${SELECTION_VAL}" ;; esac
    S1_CACHE="${S1_CACHE_ROOT}/${b}"
    S1P5_CACHE="${E2E_CACHE_ROOT}/${b}_raw"   # Stage-1.5 output only
    LOG="${LOG_DIR}/sample_stage1p5_${b}.log"
    if [[ ${SKIP_SAMPLE_STAGE1P5} -eq 1 ]]; then
        echo "[E2E] --skip-sample-stage1p5: using ${S1P5_CACHE}"
        continue
    fi
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE 1b: SAMPLE Stage-1.5 (Stage-1 cache) → ${S1P5_CACHE}"
    echo "================================================================"
    SAMPLE_CMD=("${PY}" -u scripts/stage_a_generator/sample_substitute_conds_cli.py
        --stage stage1p5 --config "${STAGE1P5_CFG}" --ckpt "${STAGE1P5_CKPT}"
        --selection-json "${sel}" --bucket "${b}" --out-dir "${S1P5_CACHE}"
        --upstream-dir "${S1_CACHE}"
        --seed "${SEED}" --cfg-scale "${CFG_SCALE}" --sampler "${SAMPLER}")
    if [[ ${DRY_RUN} -eq 1 ]]; then echo "[E2E DRY-RUN]"; echo "    \$ ${SAMPLE_CMD[*]}";
    else "${SAMPLE_CMD[@]}" 2>&1 | tee "${LOG}"; fi
done

# Phase 1c: Merge caches per-clip so the diag's --substitute-conds-dir
# has BOTH stage1_coarse (from Stage-1 cache) AND stage2_coarse_extra +
# stage2_support (from Stage-1.5 cache) in each .npz file.
for b in "${BUCKETS[@]}"; do
    S1_CACHE="${S1_CACHE_ROOT}/${b}"
    S1P5_CACHE="${E2E_CACHE_ROOT}/${b}_raw"
    MERGED="${E2E_CACHE_ROOT}/${b}"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "[E2E DRY-RUN] would merge ${S1_CACHE} + ${S1P5_CACHE} → ${MERGED}"
        continue
    fi
    echo "[E2E] PHASE 1c: merge ${S1_CACHE} + ${S1P5_CACHE} → ${MERGED}"
    "${PY}" -u -c "
import numpy as np
from pathlib import Path
s1 = Path('${S1_CACHE}'); s1p5 = Path('${S1P5_CACHE}'); out = Path('${MERGED}')
n = 0
for p in s1p5.rglob('*.npz'):
    rel = p.relative_to(s1p5)
    s1_p = s1 / rel
    if not s1_p.exists():
        raise SystemExit(f'missing Stage-1 cache for {rel}')
    s1_data = dict(np.load(s1_p))
    s1p5_data = dict(np.load(p))
    out_p = out / rel
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_p,
             stage1_coarse=s1_data['stage1_coarse'],
             stage2_coarse_extra=s1p5_data['stage2_coarse_extra'],
             stage2_support=s1p5_data['stage2_support'],
             valid_T=s1p5_data['valid_T'],
             seed=s1p5_data['seed'])
    n += 1
print(f'merged {n} clips')
"
done

# Phase 2: Run 4 Stage-2 diags with the merged substitute_conds dir.
run_diag() {
    local KIND="$1"; local BUCKET="$2"
    case "${BUCKET}" in train) sel="${SELECTION_TRAIN}" ;; val) sel="${SELECTION_VAL}" ;; esac
    case "${KIND}" in
        sustained_contact) SCRIPT="scripts/stage_b_generator/round26_sustained_contact_diag.py" ;;
        gait)              SCRIPT="scripts/stage_b_generator/round26_gait_diag.py" ;;
        body_action)       SCRIPT="scripts/stage_b_generator/round28_body_action_diag.py" ;;
        g1_soft_stance)    SCRIPT="scripts/stage_b_generator/round29_g1_soft_stance_diag.py" ;;
        *) echo "[E2E DIAG] unknown kind ${KIND}"; return 1 ;;
    esac
    MERGED="${E2E_CACHE_ROOT}/${BUCKET}"
    OUT_DIR="analyses/round31_32_end_to_end_diag/${KIND}_${BUCKET}"
    LOG="${LOG_DIR}/diag_${KIND}_${BUCKET}.log"
    mkdir -p "${OUT_DIR}"
    CMD=("${PY}" -u "${SCRIPT}"
        --config "${PB1_CFG}" --ckpt "${PB1_CKPT}"
        --selection-json "${sel}"
        --output-dir "${OUT_DIR}"
        --bucket "${BUCKET}"
        --substitute-conds-dir "${MERGED}"
        --cfg-scale "${CFG_SCALE}" --seed "${SEED}")
    echo "[E2E DIAG START] ${KIND} ${BUCKET}"
    if [[ ${DRY_RUN} -eq 1 ]]; then echo "    \$ ${CMD[*]}"; return 0; fi
    if ! "${CMD[@]}" 2>&1 | tee "${LOG}"; then
        if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
            echo "[E2E] WARN: ${KIND} ${BUCKET} failed; continuing (ALLOW_PARTIAL=1)"
        else
            echo "[E2E] FATAL: ${KIND} ${BUCKET} failed."; return 1
        fi
    fi
}

if [[ ${SKIP_DIAG} -eq 1 ]]; then
    echo "[E2E] --skip-diag: skipping Phase 2"
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PHASE 2: DIAG (4 kinds × ${#BUCKETS[@]} buckets)"
    echo "================================================================"
    for b in "${BUCKETS[@]}"; do
        for KIND in sustained_contact gait body_action g1_soft_stance; do
            run_diag "${KIND}" "${b}"
        done
    done
fi

# Phase 3: Pack.
if [[ ${DRY_RUN} -eq 0 ]]; then
    STAMP=$(date +%Y%m%d_%H%M%S)
    TARBALL="analyses/round31_32_end_to_end_results_${STAMP}.tar.gz"
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PACK -> ${TARBALL}"
    echo "================================================================"
    PACK_TARGETS=()
    for b in "${BUCKETS[@]}"; do
        for KIND in sustained_contact gait body_action g1_soft_stance; do
            D="analyses/round31_32_end_to_end_diag/${KIND}_${b}"
            [[ -d "${D}" ]] && PACK_TARGETS+=("${D}")
        done
    done
    [[ -d "${E2E_CACHE_ROOT}" ]] && PACK_TARGETS+=("${E2E_CACHE_ROOT}")
    [[ -d "${LOG_DIR}" ]] && PACK_TARGETS+=("${LOG_DIR}")
    if [[ ${#PACK_TARGETS[@]} -gt 0 ]]; then
        tar -czf "${TARBALL}" "${PACK_TARGETS[@]}"
        SIZE=$(du -h "${TARBALL}" | cut -f1)
        echo "wrote ${TARBALL}  (${SIZE})"
    else
        echo "[E2E PACK] nothing to pack"
    fi
fi

echo
echo "================================================================"
echo "End-to-end (D) diag complete."
echo "================================================================"
