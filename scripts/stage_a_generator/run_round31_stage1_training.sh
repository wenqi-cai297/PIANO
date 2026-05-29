#!/usr/bin/env bash
# Round-31 Stage-1 (Trajectory) training launcher.
#
# Per analyses/2026-05-29_stage1_and_stage1_5_design.md §"Stage-1 design".
#
# Single train variant:
#   stage1_traj_v0      23-D stage1_coarse generator
#
# Schedule: bs=64 / accum=1 / 80 ep / heldout val / val_every=5 /
# save_every=10 / warmup=500 (2× 5080). From scratch (no init_ckpt).
#
# Phase 1 (TRAIN):  the Stage-1 variant via accelerate (multi-GPU).
# Phase 2 (PACK):   tarball runs/training/stage1_*/final.pt + logs into
#                   analyses/round31_stage1_results_<stamp>.tar.gz.
#
# Downstream-coupling diagnostic is a SEPARATE script
# (`scripts/stage_a_generator/run_round31_stage1_downstream_diag.sh`,
# not yet written) that pipes Stage-1's output into frozen PB1 and
# re-runs Stage-2's 4 standard diag kinds.
#
# Usage:
#   bash scripts/stage_a_generator/run_round31_stage1_training.sh
#   bash scripts/stage_a_generator/run_round31_stage1_training.sh --dry-run
#   bash scripts/stage_a_generator/run_round31_stage1_training.sh --skip-train
#
# Environment overrides:
#   DATASETS_ROOT=...                     dataset root (default = dev Windows path)
#   ROUND31_S1_NUM_PROCESSES=N            accelerate --num_processes
#   ROUND31_S1_SINGLE_GPU=1               force single-GPU train
#   ROUND31_S1_ALLOW_PARTIAL=1            allow partial reports on failures
#   ROUND31_S1_REGEN_CONFIGS=1            force manifest/config regeneration

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
SKIP_TRAIN=0
SINGLE_GPU="${ROUND31_S1_SINGLE_GPU:-0}"
ALLOW_PARTIAL="${ROUND31_S1_ALLOW_PARTIAL:-0}"
REGEN_CONFIGS="${ROUND31_S1_REGEN_CONFIGS:-0}"

if [[ -n "${ROUND31_S1_NUM_PROCESSES:-}" ]]; then
    NUM_PROCESSES="${ROUND31_S1_NUM_PROCESSES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_PROCESSES="$(nvidia-smi -L | wc -l)"
    [[ "${NUM_PROCESSES}" -lt 1 ]] && NUM_PROCESSES=1
else
    NUM_PROCESSES=2
fi

MANIFEST="analyses/round31_stage1_manifest.json"
LOG_DIR="runs/round31_stage1"
mkdir -p "${LOG_DIR}"
if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PY="python"
    elif command -v python3 >/dev/null 2>&1; then
        PY="python3"
    else
        echo "[S1] FATAL: neither python nor python3 was found" >&2
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

echo "[S1] NUM_PROCESSES=${NUM_PROCESSES}  ALLOW_PARTIAL=${ALLOW_PARTIAL}"

# (1) Generate manifest + config if missing or stale.
GENERATOR="scripts/stage_a_generator/round31_make_stage1_configs.py"
if [[ ${REGEN_CONFIGS} -eq 1 || ! -f "${MANIFEST}" || "${GENERATOR}" -nt "${MANIFEST}" || -n "${DATASETS_ROOT:-}" ]]; then
    echo "[S1] Regenerating manifest/configs..."
    GEN_ARGS=()
    [[ -n "${DATASETS_ROOT:-}" ]] && GEN_ARGS+=(--data-root "${DATASETS_ROOT}")
    "${PY}" "${GENERATOR}" "${GEN_ARGS[@]}"
fi

# (2) Pick variants.
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
    echo "[S1] no train variants matched only='${ONLY}'"
    exit 0
fi
echo "[S1] Train variants to process:"
echo "${VARIANTS}"

# (3) Preflight.
preflight_fail=0
while IFS=' ' read -r VID CFG OUTDIR; do
    [[ -z "${VID}" ]] && continue
    if [[ ! -e "${CFG}" ]]; then
        echo "[S1 PREFLIGHT FAIL] [${VID}] missing config: ${CFG}"
        preflight_fail=1
    fi
    # Fairness: from-scratch — no init_checkpoint key.
    if [[ -e "${CFG}" ]] && grep -Eq "^[[:space:]]*init_checkpoint[[:space:]]*:" "${CFG}"; then
        echo "[S1 PREFLIGHT FAIL] [${VID}] config sets init_checkpoint (Stage-1 must train from scratch)"
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
                echo "[S1 PREFLIGHT FAIL] [${VID}] dataset root not on disk: ${br}"
            done <<< "${BAD}"
            preflight_fail=1
        fi
    done <<< "${VARIANTS}"
fi

if [[ ${preflight_fail} -ne 0 ]]; then
    echo "[S1] FATAL preflight failures."
    exit 1
fi

# (4) PHASE 1: TRAIN.
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
            TRAIN_CMD=("${PY}" -u src/piano/training/train_stage1.py --config "${CFG}")
        else
            TRAIN_CMD=(accelerate launch
                --num_processes "${NUM_PROCESSES}" --multi_gpu --mixed_precision bf16
                src/piano/training/train_stage1.py --config "${CFG}")
        fi
        if [[ ${DRY_RUN} -eq 1 ]]; then
            echo "[S1 DRY-RUN ${VID} TRAIN]"
            echo "    \$ ${TRAIN_CMD[*]}"
            TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR}"$'\n'
        else
            if "${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG}"; then
                TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR}"$'\n'
            else
                if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
                    echo "[S1] WARN: training failed for ${VID}"
                else
                    echo "[S1] FATAL: training failed for ${VID}; aborting."
                    exit 1
                fi
            fi
        fi
    else
        echo "--skip-train: skipping training for ${VID}"
        TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR}"$'\n'
    fi
done <<< "${VARIANTS}"

# (5) PHASE 2: PACK.
if [[ ${DRY_RUN} -eq 0 ]]; then
    STAMP=$(date +%Y%m%d_%H%M%S)
    TARBALL="analyses/round31_stage1_results_${STAMP}.tar.gz"
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PACK -> ${TARBALL}"
    echo "================================================================"
    PACK_TARGETS=()
    while IFS=' ' read -r VID CFG OUTDIR; do
        [[ -z "${VID}" ]] && continue
        if [[ -d "${OUTDIR}" ]]; then
            # Only pack final.pt + metrics, not all intermediate ckpts.
            FINAL="${OUTDIR}/final.pt"
            METRICS="${OUTDIR}/metrics.jsonl"
            [[ -f "${FINAL}" ]] && PACK_TARGETS+=("${FINAL}")
            [[ -f "${METRICS}" ]] && PACK_TARGETS+=("${METRICS}")
        fi
        L="${LOG_DIR}/${VID}.log"
        [[ -f "${L}" ]] && PACK_TARGETS+=("${L}")
    done <<< "${TRAINED_OK}"
    PACK_TARGETS+=("${MANIFEST}")
    [[ -f analyses/round31_stage1_manifest.md ]] && PACK_TARGETS+=("analyses/round31_stage1_manifest.md")
    if [[ ${#PACK_TARGETS[@]} -eq 0 ]]; then
        echo "[S1 PACK] nothing to pack"
    else
        tar -czf "${TARBALL}" "${PACK_TARGETS[@]}"
        SIZE=$(du -h "${TARBALL}" | cut -f1)
        echo "wrote ${TARBALL}  (${SIZE})"
    fi
fi

echo
echo "================================================================"
echo "Round-31 Stage-1 training complete."
echo "Next: bash scripts/stage_a_generator/run_round31_stage1_downstream_diag.sh"
echo "      (pipes Stage-1 output into frozen PB1 for downstream metrics)"
echo "================================================================"
