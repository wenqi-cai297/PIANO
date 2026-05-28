#!/usr/bin/env bash
# Round-29 prior-bias (PB1 AdaLN-S4) ablation launcher.
#
# Per Codex review §4 + §9 of
# analyses/2026-05-29_round29_cond_injection_prior_codex_review_for_claude_code.md
# and Phase 0 verdict in
# analyses/2026-05-29_round29_cond_usage_verdict.md.
#
# Single train variant this round:
#   r29_pb_a1_adaln_s4      A1 + AdaLN-S4 (support_walking_mean pool)
#
# Schedule: bs=32 / accum=1 / 80 ep / heldout val / val_every=5 /
# save_every=10 / warmup=250 (2× 5080). From scratch (no init_ckpt) —
# fairness requirement from the user.
#
# Phase 1 (TRAIN):  the PB1 variant via accelerate (multi-GPU).
# Phase 2 (DIAG):   sustained_contact / gait / body_action / g1_soft_stance
#                   on both train + val buckets (8 tasks).
# Phase 3 (POST-PROBE): rerun cond-usage probe on the new PB1 ckpt to
#                   check whether S4 mechanics shifted (linearity ↑ /
#                   time_shuffle/zero ↑) — separates "PB1 changed
#                   mechanism" from "PB1 found different stochastic
#                   minimum". Codex review §6 + Phase 0 verdict caveat.
# Phase 4 (PACK):   tarball everything into
#                   analyses/round29_prior_bias_results_<stamp>.tar.gz
#
# Usage:
#   bash scripts/stage_b_generator/run_round29_prior_bias_ablations.sh
#   bash scripts/stage_b_generator/run_round29_prior_bias_ablations.sh --dry-run
#   bash scripts/stage_b_generator/run_round29_prior_bias_ablations.sh --skip-train
#   bash scripts/stage_b_generator/run_round29_prior_bias_ablations.sh --skip-eval
#
# Environment overrides:
#   DATASETS_ROOT=...                     dataset root (default = dev Windows path)
#   ROUND29_PB_NUM_PROCESSES=N            accelerate --num_processes
#   ROUND29_PB_PARALLEL_DIAG_WORKERS=N    diag workers (default: NUM_PROCESSES)
#   ROUND29_PB_DIAG_CKPT_NAME=best_val.pt diag ckpt filename (default: final.pt)
#   ROUND29_PB_SINGLE_GPU=1               force single-GPU train
#   ROUND29_PB_ALLOW_PARTIAL=1            allow partial reports on failures
#   ROUND29_PB_REGEN_CONFIGS=1            force manifest/config regeneration
#   ROUND29_PB_SKIP_POST_PROBE=1          skip the Phase 3 cond-usage re-probe

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
SKIP_TRAIN=0
SKIP_EVAL=0
SKIP_POST_PROBE="${ROUND29_PB_SKIP_POST_PROBE:-0}"
SINGLE_GPU="${ROUND29_PB_SINGLE_GPU:-0}"
DIAG_CKPT_NAME="${ROUND29_PB_DIAG_CKPT_NAME:-final.pt}"
ALLOW_PARTIAL="${ROUND29_PB_ALLOW_PARTIAL:-0}"
REGEN_CONFIGS="${ROUND29_PB_REGEN_CONFIGS:-0}"

SELECTION_TRAIN="analyses/round27_tier0_train_indices_48_balanced.json"
SELECTION_VAL="analyses/round29_val_diag_indices_48_balanced.json"

A1_REF_VARIANT="r29_ns_a1_c41_s4_g1"
A1_REF_CFG="configs/training/anchordiff_${A1_REF_VARIANT}.yaml"
A1_REF_CKPT="runs/training/stageB_anchordiff_${A1_REF_VARIANT}/${DIAG_CKPT_NAME}"

if [[ -n "${ROUND29_PB_NUM_PROCESSES:-}" ]]; then
    NUM_PROCESSES="${ROUND29_PB_NUM_PROCESSES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_PROCESSES="$(nvidia-smi -L | wc -l)"
    [[ "${NUM_PROCESSES}" -lt 1 ]] && NUM_PROCESSES=1
else
    NUM_PROCESSES=2
fi
PARALLEL_DIAG_WORKERS="${ROUND29_PB_PARALLEL_DIAG_WORKERS:-${NUM_PROCESSES}}"

MANIFEST="analyses/round29_prior_bias_ablation_manifest.json"
LOG_DIR="runs/round29_prior_bias_ablation"
mkdir -p "${LOG_DIR}"
if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PY="python"
    elif command -v python3 >/dev/null 2>&1; then
        PY="python3"
    else
        echo "[PB] FATAL: neither python nor python3 was found" >&2
        exit 127
    fi
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)                  ONLY="$2"; shift 2 ;;
        --dry-run)               DRY_RUN=1; shift ;;
        --skip-train)            SKIP_TRAIN=1; shift ;;
        --skip-eval)             SKIP_EVAL=1; shift ;;
        --skip-post-probe)       SKIP_POST_PROBE=1; shift ;;
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

echo "[PB] NUM_PROCESSES=${NUM_PROCESSES}  PARALLEL_DIAG_WORKERS=${PARALLEL_DIAG_WORKERS}  DIAG_CKPT_NAME=${DIAG_CKPT_NAME}  ALLOW_PARTIAL=${ALLOW_PARTIAL}  SKIP_POST_PROBE=${SKIP_POST_PROBE}"

# (1) Generate manifest + configs if missing or stale.
GENERATOR="scripts/stage_b_generator/round29_make_prior_bias_ablation_configs.py"
if [[ ${REGEN_CONFIGS} -eq 1 || ! -f "${MANIFEST}" || "${GENERATOR}" -nt "${MANIFEST}" || -n "${DATASETS_ROOT:-}" ]]; then
    echo "[PB] Regenerating manifest/configs..."
    GEN_ARGS=()
    [[ -n "${DATASETS_ROOT:-}" ]] && GEN_ARGS+=(--data-root "${DATASETS_ROOT}")
    "${PY}" "${GENERATOR}" "${GEN_ARGS[@]}"
fi

# (2) Pick TRAIN-able variants from manifest.
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
    echo "[PB] no train variants matched only='${ONLY}'"
    exit 0
fi
echo "[PB] Train variants to process:"
echo "${VARIANTS}"

# (3) Preflight.
preflight_fail=0

while IFS=' ' read -r VID CFG OUTDIR; do
    [[ -z "${VID}" ]] && continue
    if [[ ! -e "${CFG}" ]]; then
        echo "[PB PREFLIGHT FAIL] [${VID}] missing config: ${CFG}"
        preflight_fail=1
    elif grep -q "oracle_hint" "${CFG}" 2>/dev/null; then
        echo "[PB PREFLIGHT FAIL] [${VID}] config contains dead oracle_hint fields: ${CFG}"
        preflight_fail=1
    fi
    # Fairness: PB1 must be trained from scratch. A YAML line that
    # actually sets init_checkpoint as a key (e.g. ``init_checkpoint: ...``
    # or ``  init_checkpoint:``) is forbidden; the substring inside a
    # comment is fine.
    if grep -Eq "^[[:space:]]*init_checkpoint[[:space:]]*:" "${CFG}" 2>/dev/null; then
        echo "[PB PREFLIGHT FAIL] [${VID}] config sets init_checkpoint (fairness requires from-scratch training)"
        preflight_fail=1
    fi
done <<< "${VARIANTS}"

if [[ ${SKIP_EVAL} -eq 0 ]]; then
    for sel in "${SELECTION_TRAIN}" "${SELECTION_VAL}"; do
        if [[ ! -e "${sel}" ]]; then
            echo "[PB PREFLIGHT FAIL] missing diag selection JSON: ${sel}"
            preflight_fail=1
        fi
    done
    if [[ ! -e "${A1_REF_CFG}" ]]; then
        echo "[PB PREFLIGHT FAIL] A1 reference config missing: ${A1_REF_CFG}"
        echo "    -> regenerate via:"
        echo "       python scripts/stage_b_generator/round29_make_next_step_ablation_configs.py"
        preflight_fail=1
    fi
fi

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
                echo "[PB PREFLIGHT FAIL] [${VID}] dataset root not on disk: ${br}"
            done <<< "${BAD}"
            echo "    [${VID}]   -> re-run generator with --data-root <correct path> or export DATASETS_ROOT=..."
            preflight_fail=1
        fi
    done <<< "${VARIANTS}"
fi

# Post-probe requires the A1 ckpt to be on disk (comparison baseline).
# WARN under ALLOW_PARTIAL=1, FATAL otherwise.
if [[ ${SKIP_POST_PROBE} -eq 0 && ${SKIP_EVAL} -eq 0 ]]; then
    if [[ ! -e "${A1_REF_CKPT}" ]]; then
        if [[ "${ALLOW_PARTIAL}" == "1" || ${DRY_RUN} -eq 1 ]]; then
            echo "[PB PREFLIGHT WARN] A1 reference ckpt missing (post-probe will skip comparison): ${A1_REF_CKPT}"
        else
            echo "[PB PREFLIGHT FAIL] A1 reference ckpt missing: ${A1_REF_CKPT}"
            echo "    -> server should already have it from R29-NS; if not, regenerate."
            preflight_fail=1
        fi
    fi
fi

if [[ ${preflight_fail} -ne 0 ]]; then
    echo "[PB] FATAL preflight failures."
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
            TRAIN_CMD=("${PY}" -u src/piano/training/train_anchordiff.py --config "${CFG}")
        else
            TRAIN_CMD=(accelerate launch
                --num_processes "${NUM_PROCESSES}" --multi_gpu --mixed_precision bf16
                src/piano/training/train_anchordiff.py --config "${CFG}")
        fi
        if [[ ${DRY_RUN} -eq 1 ]]; then
            echo "[PB DRY-RUN ${VID} TRAIN]"
            echo "    \$ ${TRAIN_CMD[*]}"
            TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR}"$'\n'
        else
            if "${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG}"; then
                TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR}"$'\n'
            else
                if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
                    echo "[PB] WARN: training failed for ${VID}; skipping diag"
                else
                    echo "[PB] FATAL: training failed for ${VID}; aborting."
                    exit 1
                fi
            fi
        fi
    else
        echo "--skip-train: skipping training for ${VID}"
        TRAINED_OK="${TRAINED_OK}${VID} ${CFG} ${OUTDIR}"$'\n'
    fi
done <<< "${VARIANTS}"

# (5) PHASE 2: DIAG (4 kinds × 2 buckets per variant).
if [[ ${SKIP_EVAL} -eq 1 ]]; then
    echo
    echo "--skip-eval: skipping diag"
elif [[ -z "${TRAINED_OK}" ]]; then
    echo "[PB] No variants succeeded training; no diag to run."
    if [[ "${ALLOW_PARTIAL}" != "1" ]]; then
        exit 1
    fi
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] DIAG PHASE (workers=${PARALLEL_DIAG_WORKERS})"
    echo "================================================================"

    run_diag() {
        local KIND="$1"; local VID="$2"; local CFG="$3"; local OUTDIR="$4"; local BUCKET="$5"
        local CKPT="${OUTDIR}/${DIAG_CKPT_NAME}"
        local DIAG_OUT="analyses/round29_${VID}_diag_${KIND}_${BUCKET}"
        local LOG="${LOG_DIR}/${VID}_${KIND}_${BUCKET}.log"

        if [[ ! -e "${CKPT}" ]]; then
            echo "[PB DIAG SKIP] missing ckpt for ${VID}: ${CKPT}"
            return 0
        fi
        mkdir -p "${DIAG_OUT}"

        local SEL
        if [[ "${BUCKET}" == "train" ]]; then
            SEL="${SELECTION_TRAIN}"
        else
            SEL="${SELECTION_VAL}"
        fi

        local SCRIPT
        case "${KIND}" in
            sustained_contact) SCRIPT="scripts/stage_b_generator/round26_sustained_contact_diag.py" ;;
            gait)              SCRIPT="scripts/stage_b_generator/round26_gait_diag.py" ;;
            body_action)       SCRIPT="scripts/stage_b_generator/round26_body_action_diag.py" ;;
            g1_soft_stance)    SCRIPT="scripts/stage_b_generator/round29_g1_soft_stance_diag.py" ;;
            *) echo "[PB DIAG] unknown kind ${KIND}"; return 1 ;;
        esac

        echo "[PB DIAG START] ${VID} ${KIND} ${BUCKET}"
        if [[ ${DRY_RUN} -eq 1 ]]; then
            echo "    \$ ${PY} -u ${SCRIPT} --config ${CFG} --ckpt ${CKPT} --selection-json ${SEL} --bucket ${BUCKET} --output-dir ${DIAG_OUT}"
            return 0
        fi
        "${PY}" -u "${SCRIPT}" \
            --config "${CFG}" --ckpt "${CKPT}" \
            --selection-json "${SEL}" --bucket "${BUCKET}" \
            --output-dir "${DIAG_OUT}" 2>&1 | tee "${LOG}"
    }

    # Trained variants × 4 kinds × 2 buckets.
    while IFS=' ' read -r VID CFG OUTDIR; do
        [[ -z "${VID}" ]] && continue
        for KIND in sustained_contact gait body_action g1_soft_stance; do
            for BUCKET in train val; do
                run_diag "${KIND}" "${VID}" "${CFG}" "${OUTDIR}" "${BUCKET}"
            done
        done
    done <<< "${TRAINED_OK}"
fi

# (6) PHASE 3: POST-PB1 cond-usage re-probe (per Phase 0 verdict + Codex §6).
if [[ ${SKIP_POST_PROBE} -eq 1 || ${SKIP_EVAL} -eq 1 ]]; then
    echo
    echo "--skip-post-probe or --skip-eval: skipping post-PB1 cond-usage probe"
elif [[ -z "${TRAINED_OK}" ]]; then
    echo "[PB] No variants trained; skipping post-probe."
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] POST-PB1 COND-USAGE PROBE"
    echo "================================================================"

    while IFS=' ' read -r VID CFG OUTDIR; do
        [[ -z "${VID}" ]] && continue
        CKPT="${OUTDIR}/${DIAG_CKPT_NAME}"
        OUT="analyses/round29_cond_usage_probe/${VID}_val"
        LOG="${LOG_DIR}/${VID}_post_probe.log"
        if [[ ! -e "${CKPT}" ]]; then
            echo "[PB POST-PROBE SKIP] missing ckpt for ${VID}: ${CKPT}"
            continue
        fi
        mkdir -p "${OUT}"
        echo "[PB POST-PROBE] ${VID} -> ${OUT}"
        if [[ ${DRY_RUN} -eq 1 ]]; then
            echo "    \$ ${PY} -u scripts/stage_b_generator/round29_cond_usage_probe.py --config ${CFG} --ckpt ${CKPT} --selection-json ${SELECTION_VAL} --bucket val --output-dir ${OUT} --variant-id ${VID} --cfg-scale 1.0 --seed 42"
            continue
        fi
        "${PY}" -u scripts/stage_b_generator/round29_cond_usage_probe.py \
            --config "${CFG}" --ckpt "${CKPT}" \
            --selection-json "${SELECTION_VAL}" \
            --output-dir "${OUT}" \
            --bucket val \
            --variant-id "${VID}" \
            --cfg-scale 1.0 --seed 42 2>&1 | tee "${LOG}"
    done <<< "${TRAINED_OK}"
fi

# (7) PHASE 4: PACK.
if [[ ${DRY_RUN} -eq 0 ]]; then
    STAMP=$(date +%Y%m%d_%H%M%S)
    TARBALL="analyses/round29_prior_bias_results_${STAMP}.tar.gz"
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] PACK -> ${TARBALL}"
    echo "================================================================"
    PACK_TARGETS=()
    while IFS=' ' read -r VID CFG OUTDIR; do
        [[ -z "${VID}" ]] && continue
        for KIND in sustained_contact gait body_action g1_soft_stance; do
            for BUCKET in train val; do
                D="analyses/round29_${VID}_diag_${KIND}_${BUCKET}"
                [[ -d "${D}" ]] && PACK_TARGETS+=("${D}")
            done
        done
        P="analyses/round29_cond_usage_probe/${VID}_val"
        [[ -d "${P}" ]] && PACK_TARGETS+=("${P}")
        L="${LOG_DIR}/${VID}.log"
        [[ -f "${L}" ]] && PACK_TARGETS+=("${L}")
    done <<< "${TRAINED_OK}"
    PACK_TARGETS+=("${MANIFEST}")
    [[ -f analyses/round29_prior_bias_ablation_manifest.md ]] && PACK_TARGETS+=("analyses/round29_prior_bias_ablation_manifest.md")
    if [[ ${#PACK_TARGETS[@]} -eq 0 ]]; then
        echo "[PB PACK] nothing to pack"
    else
        tar -czf "${TARBALL}" "${PACK_TARGETS[@]}"
        SIZE=$(du -h "${TARBALL}" | cut -f1)
        echo "wrote ${TARBALL}  (${SIZE})"
    fi
fi

echo
echo "================================================================"
echo "Round-29 prior-bias ablation complete."
echo "scp back:  scp <server>:$(pwd)/${TARBALL:-analyses/round29_prior_bias_results_*.tar.gz} ./analyses/"
echo "================================================================"
