#!/usr/bin/env bash
# Round-32 Stage-1.5 (Interaction Plan) training launcher.
#
# Per analyses/2026-05-29_stage1_and_stage1_5_design.md §"Stage-1.5 design".
#
# Single train variant:
#   stage1p5_interaction_v0      31-D (C41 18 + S4 13) generator
#
# Schedule: bs=48 / accum=1 / 80 ep / heldout val / val_every=5 /
# save_every=10 / warmup=500 (2× 5080). From scratch.
#
# Phase 1 (TRAIN), Phase 2 (PACK).
#
# Downstream-coupling diagnostic is a SEPARATE script
# (`run_round32_stage1p5_downstream_diag.sh`, not yet written) that pipes
# Stage-1.5 output into frozen PB1 with oracle Stage-1 cond, then runs
# Stage-2's 4 standard diag kinds.

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
SKIP_TRAIN=0
SINGLE_GPU="${ROUND32_S1P5_SINGLE_GPU:-0}"
ALLOW_PARTIAL="${ROUND32_S1P5_ALLOW_PARTIAL:-0}"
REGEN_CONFIGS="${ROUND32_S1P5_REGEN_CONFIGS:-0}"

if [[ -n "${ROUND32_S1P5_NUM_PROCESSES:-}" ]]; then
    NUM_PROCESSES="${ROUND32_S1P5_NUM_PROCESSES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_PROCESSES="$(nvidia-smi -L | wc -l)"
    [[ "${NUM_PROCESSES}" -lt 1 ]] && NUM_PROCESSES=1
else
    NUM_PROCESSES=2
fi

MANIFEST="analyses/round32_stage1p5_manifest.json"
LOG_DIR="runs/round32_stage1p5"
mkdir -p "${LOG_DIR}"
if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PY="python"
    elif command -v python3 >/dev/null 2>&1; then
        PY="python3"
    else
        echo "[S1.5] FATAL: neither python nor python3 was found" >&2
        exit 127
    fi
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)          ONLY="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --skip-train)    SKIP_TRAIN=1; shift ;;
        --num-processes) NUM_PROCESSES="$2"; shift 2 ;;
        --single-gpu)    SINGLE_GPU=1; shift ;;
        --regen-configs) REGEN_CONFIGS=1; shift ;;
        -h|--help)
            sed -n '1,40p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

echo "[S1.5] NUM_PROCESSES=${NUM_PROCESSES}  ALLOW_PARTIAL=${ALLOW_PARTIAL}"

GENERATOR="scripts/stage_a_generator/round32_make_stage1p5_configs.py"
if [[ ${REGEN_CONFIGS} -eq 1 || ! -f "${MANIFEST}" || "${GENERATOR}" -nt "${MANIFEST}" || -n "${DATASETS_ROOT:-}" ]]; then
    echo "[S1.5] Regenerating manifest/configs..."
    GEN_ARGS=()
    [[ -n "${DATASETS_ROOT:-}" ]] && GEN_ARGS+=(--data-root "${DATASETS_ROOT}")
    "${PY}" "${GENERATOR}" "${GEN_ARGS[@]}"
fi

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
    echo "[S1.5] no train variants matched only='${ONLY}'"
    exit 0
fi
echo "[S1.5] Train variants to process:"
echo "${VARIANTS}"

# Preflight.
preflight_fail=0
while IFS=' ' read -r VID CFG OUTDIR; do
    [[ -z "${VID}" ]] && continue
    if [[ ! -e "${CFG}" ]]; then
        echo "[S1.5 PREFLIGHT FAIL] [${VID}] missing config: ${CFG}"
        preflight_fail=1
    fi
    if [[ -e "${CFG}" ]] && grep -Eq "^[[:space:]]*init_checkpoint[[:space:]]*:" "${CFG}"; then
        echo "[S1.5 PREFLIGHT FAIL] [${VID}] config sets init_checkpoint (Stage-1.5 must train from scratch)"
        preflight_fail=1
    fi
done <<< "${VARIANTS}"

if [[ ${DRY_RUN} -eq 0 && ${SKIP_TRAIN} -eq 0 ]]; then
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
                echo "[S1.5 PREFLIGHT FAIL] [${VID}] dataset root not on disk: ${br}"
            done <<< "${BAD}"
            preflight_fail=1
        fi
    done <<< "${VARIANTS}"
fi

if [[ ${preflight_fail} -ne 0 ]]; then
    echo "[S1.5] FATAL preflight failures."
    exit 1
fi

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
            TRAIN_CMD=("${PY}" -u src/piano/training/train_stage1p5.py --config "${CFG}")
        else
            TRAIN_CMD=(accelerate launch
                --num_processes "${NUM_PROCESSES}" --multi_gpu --mixed_precision bf16
                src/piano/training/train_stage1p5.py --config "${CFG}")
        fi
        if [[ ${DRY_RUN} -eq 1 ]]; then
            echo "[S1.5 DRY-RUN ${VID} TRAIN]"
            echo "    \$ ${TRAIN_CMD[*]}"
            TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR}"$'\n'
        else
            if "${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG}"; then
                TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR}"$'\n'
            else
                if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
                    echo "[S1.5] WARN: training failed for ${VID}"
                else
                    echo "[S1.5] FATAL: training failed for ${VID}; aborting."
                    exit 1
                fi
            fi
        fi
    else
        echo "--skip-train: skipping training for ${VID}"
        TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR}"$'\n'
    fi
done <<< "${VARIANTS}"

if [[ ${DRY_RUN} -eq 0 ]]; then
    STAMP=$(date +%Y%m%d_%H%M%S)
    TARBALL="analyses/round32_stage1p5_results_${STAMP}.tar.gz"
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PACK -> ${TARBALL}"
    echo "================================================================"
    PACK_TARGETS=()
    while IFS=' ' read -r VID CFG OUTDIR; do
        [[ -z "${VID}" ]] && continue
        if [[ -d "${OUTDIR}" ]]; then
            FINAL="${OUTDIR}/final.pt"
            METRICS="${OUTDIR}/metrics.jsonl"
            [[ -f "${FINAL}" ]] && PACK_TARGETS+=("${FINAL}")
            [[ -f "${METRICS}" ]] && PACK_TARGETS+=("${METRICS}")
        fi
        L="${LOG_DIR}/${VID}.log"
        [[ -f "${L}" ]] && PACK_TARGETS+=("${L}")
    done <<< "${TRAINED_OK}"
    PACK_TARGETS+=("${MANIFEST}")
    [[ -f analyses/round32_stage1p5_manifest.md ]] && PACK_TARGETS+=("analyses/round32_stage1p5_manifest.md")
    if [[ ${#PACK_TARGETS[@]} -eq 0 ]]; then
        echo "[S1.5 PACK] nothing to pack"
    else
        tar -czf "${TARBALL}" "${PACK_TARGETS[@]}"
        SIZE=$(du -h "${TARBALL}" | cut -f1)
        echo "wrote ${TARBALL}  (${SIZE})"
    fi
fi

echo
echo "================================================================"
echo "Round-32 Stage-1.5 training complete."
echo "================================================================"
