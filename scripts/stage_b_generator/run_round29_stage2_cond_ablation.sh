#!/usr/bin/env bash
# Round-29 Stage-2 condition + injection ablation launcher (Bash).
#
# Mirror of run_round29_stage2_cond_ablation.py for Linux servers. Both
# launchers must stay behavior-equivalent (Codex post-review §P1/§P2).
#
# Usage:
#   bash scripts/stage_b_generator/run_round29_stage2_cond_ablation.sh --group injection
#   bash scripts/stage_b_generator/run_round29_stage2_cond_ablation.sh --group content
#   bash scripts/stage_b_generator/run_round29_stage2_cond_ablation.sh --only r29_a0_input_add
#   bash scripts/stage_b_generator/run_round29_stage2_cond_ablation.sh --group A_injection --dry-run
#
# Group aliases:
#   injection -> A_injection
#   coarse    -> B_coarse
#   interaction -> C_interaction
#   support   -> D_support
#   body      -> E_body
#   final     -> F_final
#   content   -> B_coarse,C_interaction,D_support,E_body
#   all       -> all
#
# Environment overrides:
#   ROUND29_SINGLE_GPU=1            run on a single GPU instead of accelerate launch
#   ROUND29_DIAG_CKPT_NAME=best_val.pt  diagnostic checkpoint filename
#   ROUND29_INIT_CKPT=...           init checkpoint path (when regenerating configs)
#
# Reviewed prompt section P1+P2 before fixing this launcher: yes.

set -euo pipefail
cd "$(dirname "$0")/../.."

GROUP="all"
ONLY=""
DRY_RUN=0
SKIP_TRAIN=0
SKIP_EVAL=0
SKIP_PREFLIGHT=0
ALLOW_MISSING_DIAG=0
SINGLE_GPU="${ROUND29_SINGLE_GPU:-0}"
DIAG_CKPT_NAME="${ROUND29_DIAG_CKPT_NAME:-final.pt}"
MANIFEST="analyses/round29_stage2_cond_ablation_manifest.json"
LOG_DIR="runs/round29_stage2_cond_ablation"
mkdir -p "${LOG_DIR}"

PY="${PY:-python}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --group)         GROUP="$2"; shift 2 ;;
        --only)          ONLY="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --skip-train)    SKIP_TRAIN=1; shift ;;
        --skip-eval)     SKIP_EVAL=1; shift ;;
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        --allow-missing-diag-inputs) ALLOW_MISSING_DIAG=1; shift ;;
        --diag-ckpt-name) DIAG_CKPT_NAME="$2"; shift 2 ;;
        --single-gpu)    SINGLE_GPU=1; shift ;;
        -h|--help)
            sed -n '1,33p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Resolve group aliases to a comma-separated manifest-group string.
resolve_group() {
    case "$1" in
        injection)   echo "A_injection" ;;
        coarse)      echo "B_coarse" ;;
        interaction) echo "C_interaction" ;;
        support)     echo "D_support" ;;
        body)        echo "E_body" ;;
        final)       echo "F_final" ;;
        content)     echo "B_coarse,C_interaction,D_support,E_body" ;;
        all)         echo "all" ;;
        A_injection|B_coarse|C_interaction|D_support|E_body|F_final) echo "$1" ;;
        *)
            echo "ERROR: unknown --group value '$1'" >&2
            exit 2 ;;
    esac
}

GROUPS_RESOLVED="$(resolve_group "${GROUP}")"

# Generate manifest if missing.
if [[ ! -f "${MANIFEST}" ]]; then
    echo "[R29] Manifest missing — running config generator..."
    GEN_ARGS=()
    [[ -n "${ROUND29_INIT_CKPT:-}" ]] && GEN_ARGS+=(--init-checkpoint "${ROUND29_INIT_CKPT}")
    [[ -n "${DATASETS_ROOT:-}" ]]     && GEN_ARGS+=(--data-root "${DATASETS_ROOT}")
    "${PY}" scripts/stage_b_generator/round29_make_stage2_cond_ablation_configs.py "${GEN_ARGS[@]}"
fi

# Pick variants via Python helper (reads manifest, applies group filter).
PICK_SCRIPT='
import json, sys
m = json.load(open(sys.argv[1]))
groups = sys.argv[2]
only = sys.argv[3]
want_groups = None if groups == "all" else set(groups.split(","))
want_only = set(only.split(",")) if only else None
for v in m["variants"]:
    if want_only is not None:
        if v["variant_id"] not in want_only: continue
    elif want_groups is not None and v["group"] not in want_groups:
        continue
    # Six fields: vid grp cfg outdir train_subset diag_selection init_ckpt
    diag_sel = v.get("diag_selection_file") or v["subset_file"]
    print(v["variant_id"], v["group"], v["config_path"], v["output_dir"], v["subset_file"], diag_sel, v.get("init_checkpoint",""))
'
VARIANTS="$("${PY}" -c "${PICK_SCRIPT}" "${MANIFEST}" "${GROUPS_RESOLVED}" "${ONLY}")"

if [[ -z "${VARIANTS}" ]]; then
    echo "[R29] no variants matched group='${GROUP}' only='${ONLY}'"
    exit 0
fi

echo "[R29] Variants to process:"
echo "${VARIANTS}"

# Preflight.
preflight_fail=0
if [[ ${SKIP_PREFLIGHT} -eq 0 && ${DRY_RUN} -eq 0 ]]; then
    echo "[R29] Preflight..."
    while IFS=' ' read -r VID GRP CFG OUTDIR SUBSET DIAG_SEL INIT_CKPT; do
        [[ -z "${VID}" ]] && continue
        for p in "${CFG}" "${SUBSET}" "${DIAG_SEL}"; do
            if [[ ! -e "${p}" ]]; then
                echo "    [${VID}] missing: ${p}"
                preflight_fail=1
            fi
        done
        # Diag selection JSON must have non-empty selected/candidates.
        if [[ -e "${DIAG_SEL}" && ${SKIP_EVAL} -eq 0 ]]; then
            N_SEL="$("${PY}" -c "
import json, sys
data = json.load(open(sys.argv[1]))
sel = data.get('selected') or data.get('candidates') or []
print(len(sel))
" "${DIAG_SEL}")"
            if [[ "${N_SEL}" == "0" ]]; then
                echo "    [${VID}] diag selection JSON has empty selected list: ${DIAG_SEL}"
                echo "    [${VID}]   -> looks like train-indices (int positions); diag needs eval-selection (list of {subset, seq_id})."
                preflight_fail=1
            fi
        fi
        if [[ ${SKIP_TRAIN} -eq 0 && -n "${INIT_CKPT}" && ! -e "${INIT_CKPT}" ]]; then
            echo "    [${VID}] init_checkpoint not on disk: ${INIT_CKPT}"
            preflight_fail=1
        fi
        # Dataset roots — parse the YAML and verify each root exists.
        # Skipping this lets training fail later at FileNotFoundError.
        if [[ ${SKIP_TRAIN} -eq 0 && -e "${CFG}" ]]; then
            BAD_ROOTS="$("${PY}" -c "
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(open(sys.argv[1]))
for ds in (cfg.get('data', {}).get('datasets') or []):
    root = ds.get('root', '')
    if root and not Path(root).exists():
        print(f\"{ds.get('name')}={root}\")
" "${CFG}")"
            if [[ -n "${BAD_ROOTS}" ]]; then
                while IFS= read -r br; do
                    echo "    [${VID}] dataset root not on disk: ${br}"
                done <<< "${BAD_ROOTS}"
                echo "    [${VID}]   -> re-run generator with --data-root <correct path> or export DATASETS_ROOT=..."
                preflight_fail=1
            fi
        fi
        if [[ ${SKIP_TRAIN} -eq 1 && ${SKIP_EVAL} -eq 0 ]]; then
            DIAG_CKPT="${OUTDIR}/${DIAG_CKPT_NAME}"
            if [[ ! -e "${DIAG_CKPT}" && ${ALLOW_MISSING_DIAG} -eq 0 ]]; then
                echo "    [${VID}] diag ckpt missing: ${DIAG_CKPT}"
                preflight_fail=1
            fi
        fi
    done <<< "${VARIANTS}"
    if [[ ${preflight_fail} -ne 0 ]]; then
        echo "[R29] FATAL preflight failures. Fix them or pass --skip-preflight."
        exit 1
    fi
fi

# Smoke test (fast, dry-run mode).
if [[ ${DRY_RUN} -eq 0 ]]; then
    echo "[R29] Smoke test..."
    "${PY}" scripts/stage_b_generator/round29_stage2_cond_smoke_test.py --dry-run
fi

# Resolve bucket per selection JSON.
selection_bucket() {
    local sel="$1"
    if [[ ! -e "${sel}" ]]; then echo "train"; return; fi
    "${PY}" -c "
import json, sys
data = json.load(open(sys.argv[1]))
b = data.get('bucket', 'train')
if b not in ('train','val'): b = 'train'
print(b)" "${sel}"
}

# Per-variant train + diag.
while IFS=' ' read -r VID GRP CFG OUTDIR SUBSET DIAG_SEL INIT_CKPT; do
    [[ -z "${VID}" ]] && continue
    LOG="${LOG_DIR}/${VID}.log"
    BUCKET="$(selection_bucket "${DIAG_SEL}")"
    CKPT_PATH="${OUTDIR}/${DIAG_CKPT_NAME}"

    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] BEGIN ${VID}  group=${GRP}"
    echo "    config:        ${CFG}"
    echo "    output:        ${OUTDIR}"
    echo "    train subset:  ${SUBSET}"
    echo "    diag selection: ${DIAG_SEL}  bucket=${BUCKET}"
    echo "    log:           ${LOG}"
    echo "================================================================"

    # TRAIN
    if [[ ${SKIP_TRAIN} -eq 0 ]]; then
        if [[ ${DRY_RUN} -eq 1 ]]; then
            if [[ "${SINGLE_GPU}" == "1" ]]; then
                echo "[R29 DRY-RUN ${VID} TRAIN]"
                echo "    \$ ${PY} -u src/piano/training/train_anchordiff.py --config ${CFG}"
            else
                echo "[R29 DRY-RUN ${VID} TRAIN]"
                echo "    \$ accelerate launch --num_processes 2 --multi_gpu --mixed_precision bf16 src/piano/training/train_anchordiff.py --config ${CFG}"
            fi
        else
            if [[ "${SINGLE_GPU}" == "1" ]]; then
                "${PY}" -u src/piano/training/train_anchordiff.py --config "${CFG}" 2>&1 | tee -a "${LOG}"
            else
                accelerate launch \
                    --num_processes 2 --multi_gpu --mixed_precision bf16 \
                    src/piano/training/train_anchordiff.py --config "${CFG}" 2>&1 | tee -a "${LOG}"
            fi
        fi
    else
        echo "--skip-train: skipping training for ${VID}"
    fi

    # DIAGNOSTICS — call with the actual diag CLI.
    if [[ ${SKIP_EVAL} -eq 0 ]]; then
        for kind in sustained_contact gait body_action; do
            case "${kind}" in
                sustained_contact) DIAG_SCRIPT="scripts/stage_b_generator/round26_sustained_contact_diag.py" ;;
                gait)              DIAG_SCRIPT="scripts/stage_b_generator/round26_gait_diag.py" ;;
                body_action)       DIAG_SCRIPT="scripts/stage_b_generator/round28_body_action_diag.py" ;;
            esac
            OUT_DIR="analyses/round29_${VID}_diag_${kind}"
            CMD=("${PY}" -u "${DIAG_SCRIPT}" \
                 --config "${CFG}" \
                 --ckpt "${CKPT_PATH}" \
                 --selection-json "${DIAG_SEL}" \
                 --output-dir "${OUT_DIR}" \
                 --bucket "${BUCKET}")
            if [[ ${DRY_RUN} -eq 1 ]]; then
                echo "[R29 DRY-RUN ${VID} DIAG/${kind}]"
                echo "    \$ ${CMD[*]}"
                continue
            fi
            if [[ ! -e "${CKPT_PATH}" && ${ALLOW_MISSING_DIAG} -eq 0 ]]; then
                echo "[R29] FATAL: diag ckpt missing for ${VID}/${kind}: ${CKPT_PATH}"
                echo "      Pass --allow-missing-diag-inputs to skip."
                exit 2
            fi
            mkdir -p "${OUT_DIR}"
            "${CMD[@]}" 2>&1 | tee -a "${LOG}" || true
        done
    else
        echo "--skip-eval: skipping diag for ${VID}"
    fi
done <<< "${VARIANTS}"

# Summarize.
if [[ ${DRY_RUN} -eq 0 ]]; then
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] Summarizing..."
    "${PY}" -u scripts/stage_b_generator/round29_summarize_stage2_cond_ablation.py \
        --manifest "${MANIFEST}" \
        --output-json analyses/round29_stage2_cond_ablation_summary.json \
        --output-md   analyses/round29_stage2_cond_ablation_summary.md \
        --allow-missing-results \
        2>&1 | tee -a "${LOG_DIR}/summary.log"
fi

# Pack.
if [[ ${DRY_RUN} -eq 0 ]]; then
    STAMP="$(date +%Y%m%d_%H%M%S)"
    PACK="analyses/round29_results_${STAMP}.tar.gz"
    tar -czf "${PACK}" \
        analyses/round29_stage2_cond_ablation_manifest.* \
        analyses/round29_stage2_cond_ablation_summary.* \
        analyses/round29_*_diag_* 2>/dev/null || true
    echo "[R29] Packed ${PACK}"
fi
