#!/usr/bin/env bash
# Round-28 Stage-2 oracle interface refinement: train + eval selected
# variants from analyses/round28_claude_code_stage2_oracle_interface_prompt.md.
#
# Default protocol runs only Group A. Run A0/A1/A2 first, inspect metrics,
# then launch A3/B/C with ROUND28_BEST_INJECTION_MODE set to the measured
# winner (gated_input or input_add+per_layer_adapter).
#
# Variants:
#   a0  reproduce T0-A3 (input_add)              ~20 min
#   a1  interaction hint + gated_input            ~20 min
#   a2  interaction hint + input_add+adapter      ~20 min
#   a3  best Group A, 1000 epochs                 ~60 min
#   b0  baseline (no hints)                       ~15 min
#   b1  interaction hint only (best injection)    ~20 min
#   b2  body-action hint only, all_on mask        ~20 min
#   b3  body-action hint only, energy mask        ~20 min
#   b4  interaction + body-action hints           ~25 min
#   c1  best hints + gait losses                  ~25 min
#   c2  best hints + small hint-contact consist.  ~25 min
#   c3  best hints + gait + hint-contact consist. ~25 min
#
# Each variant:
#   1. TRAIN  48-clip overfit, 300 epochs (1000 for a3), warm-start from v27.
#   2. EVAL   sustained-contact + gait + body-action diagnostics on the
#             same 48 clips on best_val.pt and final.pt.
#   3. PACK   tarball (no .pt ckpts).
#
# Usage:
#   bash scripts/stage_b_generator/run_round28_train.sh                    # A0/A1/A2 only
#   bash scripts/stage_b_generator/run_round28_train.sh r28_a0_input_add   # one
#   bash scripts/stage_b_generator/run_round28_train.sh r28_a0_input_add r28_a1_gated_input r28_a2_per_layer_adapter
#   ROUND28_BEST_INJECTION_MODE=gated_input bash scripts/stage_b_generator/run_round28_train.sh r28_a3_best_long r28_b0_baseline ...
#   ROUND28_BEST_INJECTION_MODE=per_layer_adapter bash scripts/stage_b_generator/run_round28_train.sh r28_a3_best_long r28_b2_body_only_all_on
#
# Skip-stage:
#   ROUND28_RESUME_FROM=eval  bash <this>   # skip PREP + TRAIN
#   ROUND28_RESUME_FROM=pack  bash <this>   # only pack
#
# Single-GPU fallback:
#   ROUND28_SINGLE_GPU=1 bash <this>
#
set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="runs/round28_train"
mkdir -p "${LOG_DIR}"

V27_CKPT="runs/training/stageB_anchordiff_v27_stage2_anchoraware_FULL_DATA/final.pt"
V27_CFG_LOCAL="configs/training/anchordiff_v27_stage2_anchoraware_FULL_DATA_local.yaml"
TRAIN_INDICES="${ROUND28_TRAIN_INDICES:-analyses/round27_tier0_train_indices_48_balanced.json}"
BODY_INDICES="${ROUND28_BODY_ACTION_INDICES:-analyses/round28_body_action_train_indices_48.json}"
BEST_INJECTION_MODE="${ROUND28_BEST_INJECTION_MODE:-}"

SINGLE_GPU="${ROUND28_SINGLE_GPU:-0}"
RESUME_FROM="${ROUND28_RESUME_FROM:-}"

DEFAULT_VARIANTS=(
    r28_a0_input_add
    r28_a1_gated_input
    r28_a2_per_layer_adapter
    r28_a2b_adapter_only
)

FOLLOWUP_VARIANTS=(
    r28_a3_best_long
    r28_b0_baseline
    r28_b1_interaction_only
    r28_b2_body_only_all_on
    r28_b3_body_only_energy
    r28_b4_interaction_plus_body
    r28_c1_hints_plus_gait
    r28_c2_hints_plus_hint_consistency
    r28_c3_hints_gait_consistency
)

if [[ $# -gt 0 ]]; then
    VARIANTS=("$@")
else
    VARIANTS=("${DEFAULT_VARIANTS[@]}")
fi

_should_skip() {
    local stages=(prep train eval pack)
    [[ -z "${RESUME_FROM}" ]] && return 1
    local target_idx=-1 current_idx=-1 i
    for ((i=0; i<${#stages[@]}; i++)); do
        [[ "${stages[i]}" == "$1" ]] && current_idx=$i
        [[ "${stages[i]}" == "${RESUME_FROM}" ]] && target_idx=$i
    done
    [[ $target_idx -lt 0 ]] && { echo "WARN: unknown ROUND28_RESUME_FROM=${RESUME_FROM}"; return 1; }
    [[ $current_idx -lt $target_idx ]]
}

run_step() {
    local NAME="$1"; shift
    local LOG="${LOG_DIR}/${NAME}.log"
    local T0
    T0=$(date +%s)
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] START ${NAME}"
    echo "    log: ${LOG}"
    echo "================================================================"
    PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$@" 2>&1 | tee "${LOG}"
    local T1
    T1=$(date +%s)
    echo "[$(date '+%F %T')] DONE ${NAME} in $((T1-T0))s"
}

_cfg_local() {
    echo "configs/training/anchordiff_$1_48clip_local.yaml"
}
_run_dir() {
    echo "runs/training/stageB_anchordiff_$1_48clip"
}
_is_followup_variant() {
    local V="$1"
    case "${V}" in
        r28_a3_*|r28_b*|r28_c*) return 0 ;;
        *) return 1 ;;
    esac
}
_uses_body_subset() {
    local V="$1"
    case "${V}" in
        r28_b*) return 0 ;;
        *) return 1 ;;
    esac
}
_any_followup_variant() {
    local V
    for V in "${VARIANTS[@]}"; do
        if _is_followup_variant "${V}"; then
            return 0
        fi
    done
    return 1
}
_any_body_subset_variant() {
    local V
    for V in "${VARIANTS[@]}"; do
        if _uses_body_subset "${V}"; then
            return 0
        fi
    done
    return 1
}

if _any_followup_variant && [[ -z "${BEST_INJECTION_MODE}" ]]; then
    echo "ERROR: A3/B/C variants require ROUND28_BEST_INJECTION_MODE=gated_input | per_layer_adapter | adapter_only."
    echo "Note: per_layer_adapter = input_add + per-layer adapters; adapter_only = pure per-layer adapters (A2b)."
    echo "Run A0/A1/A2/A2b first, inspect metrics, then rerun follow-up variants with the winner set."
    exit 1
fi

# ============================================================
# PREP
# ============================================================
if _should_skip prep; then
    echo "[SKIP] PREP (ROUND28_RESUME_FROM=${RESUME_FROM})"
else
    # Re-generate R28 configs from the make-configs script in case the
    # YAML drifted from the variant-table source-of-truth.
    run_step "prep_make_configs" \
        conda run --no-capture-output -n piano python -u \
            scripts/stage_b_generator/round28_make_configs.py \
            --best-injection-mode "${BEST_INJECTION_MODE:-per_layer_adapter}" \
            --balanced-subset-file "${TRAIN_INDICES}" \
            --body-action-subset-file "${BODY_INDICES}"

    run_step "prep_make_local_configs" \
        bash scripts/stage_b_generator/run_round28_make_local_configs.sh

    # v27 local config needed for baseline eval below.
    if [[ ! -f "${V27_CFG_LOCAL}" ]]; then
        run_step "prep_v27_local_for_baseline" \
            bash scripts/stage_b_generator/run_round26_make_local_configs.sh
    fi

    if [[ ! -f "${TRAIN_INDICES}" ]]; then
        run_step "build_train_indices" \
            conda run --no-capture-output -n piano python -u \
                scripts/stage_b_generator/round27_build_tier0_train_indices.py \
                --config "${V27_CFG_LOCAL}" \
                --n-clips 48 \
                --output "${TRAIN_INDICES}" \
                --max-candidates-per-subset 600
    fi

    if _any_body_subset_variant && [[ ! -f "${BODY_INDICES}" ]]; then
        run_step "build_body_action_indices" \
            conda run --no-capture-output -n piano python -u \
                scripts/stage_b_generator/round28_build_body_action_subset.py \
                --config "${V27_CFG_LOCAL}" \
                --n-clips 48 \
                --output "${BODY_INDICES}" \
                --max-candidates-per-subset 900
    fi
fi

# ---------- preflight ----------
for V in "${VARIANTS[@]}"; do
    CFG_LOCAL="$(_cfg_local "$V")"
    [[ -f "${CFG_LOCAL}" ]] || { echo "ERROR: missing config: ${CFG_LOCAL}"; exit 1; }
done
[[ -f "${V27_CKPT}" ]] || { echo "ERROR: missing v27 ckpt: ${V27_CKPT}"; exit 1; }
[[ -f "${TRAIN_INDICES}" ]] || { echo "ERROR: missing train indices: ${TRAIN_INDICES}"; exit 1; }
if _any_body_subset_variant; then
    [[ -f "${BODY_INDICES}" ]] || { echo "ERROR: missing body-action indices: ${BODY_INDICES}"; exit 1; }
fi
[[ -d "cache/stage1_coarse_v1_full" ]] || { echo "ERROR: missing Stage-1 cache"; exit 1; }

# ============================================================
# TRAIN
# ============================================================
if _should_skip train; then
    echo "[SKIP] TRAIN (ROUND28_RESUME_FROM=${RESUME_FROM})"
else
    for V in "${VARIANTS[@]}"; do
        CFG_LOCAL="$(_cfg_local "$V")"

        echo "================================================================"
        echo "[$(date '+%F %T')] TRAIN ${V}"
        echo "    config: ${CFG_LOCAL}"
        echo "================================================================"

        if [[ "${SINGLE_GPU}" == "1" ]]; then
            run_step "${V}_train_single_gpu" \
                env CUDA_VISIBLE_DEVICES=0 \
                conda run --no-capture-output -n piano accelerate launch \
                    --num_processes 1 --mixed_precision bf16 \
                    --main_process_port 29500 \
                    src/piano/training/train_anchordiff.py \
                    --config "${CFG_LOCAL}"
        else
            run_step "${V}_train_ddp" \
                conda run --no-capture-output -n piano accelerate launch \
                    --num_processes 2 --gpu_ids 0,1 --mixed_precision bf16 \
                    --main_process_port 29500 \
                    src/piano/training/train_anchordiff.py \
                    --config "${CFG_LOCAL}"
        fi
    done
fi

# ============================================================
# EVAL: sustained-contact + gait + body-action on best_val.pt and final.pt
# ============================================================
EVAL_SUBSET_BALANCED="${ROUND28_EVAL_SUBSET_BALANCED:-analyses/round27_tier0_eval_selection_balanced.json}"
EVAL_SUBSET_BODY="${ROUND28_EVAL_SUBSET_BODY:-analyses/round28_body_action_eval_selection.json}"

if _should_skip eval; then
    echo "[SKIP] EVAL (ROUND28_RESUME_FROM=${RESUME_FROM})"
else
    build_eval_selection() {
        local SRC_INDICES="$1"
        local OUT_JSON="$2"
        local DESC="$3"
        [[ -f "${OUT_JSON}" ]] && return 0
        echo "[$(date '+%F %T')] Building eval selection JSON: ${OUT_JSON}"
        conda run --no-capture-output -n piano python -c "
import json
src = json.load(open('${SRC_INDICES}', encoding='utf-8'))
clips = src['clips']
out = {
    'description': '${DESC}',
    'selection_source': '${SRC_INDICES}',
    'bucket': 'train',
    'n_clips': len(clips),
    'selected': [
        {'subset': c['subset'], 'seq_id': c['seq_id'],
         'mode_category': c.get('mode_category', c.get('body_action_category', 'unknown')),
         'text': c.get('text', ''), 'confidence': 1.0, 'n_known_valid_modes': 1}
        for c in clips
    ],
}
json.dump(out, open('${OUT_JSON}', 'w', encoding='utf-8'), indent=2)
print(f'wrote {len(clips)} clips to ${OUT_JSON}')
"
    }

    build_eval_selection \
        "${TRAIN_INDICES}" "${EVAL_SUBSET_BALANCED}" \
        "Round-28 balanced train-bucket overfit selection."
    if _any_body_subset_variant; then
        build_eval_selection \
            "${BODY_INDICES}" "${EVAL_SUBSET_BODY}" \
            "Round-28 body-action train-bucket overfit selection."
    fi

    run_diagnostics() {
        local NAME="$1"; shift
        local CFG_LOCAL="$1"; shift
        local CKPT="$1"; shift
        local OUT_DIR="$1"; shift
        local SELECTION_JSON="$1"; shift
        mkdir -p "${OUT_DIR}"

        run_step "${NAME}_sustained" \
            conda run --no-capture-output -n piano python -u \
                scripts/stage_b_generator/round26_sustained_contact_diag.py \
                --config "${CFG_LOCAL}" --ckpt "${CKPT}" \
                --selection-json "${SELECTION_JSON}" \
                --output-dir "${OUT_DIR}" \
                --bucket train --cfg-scale 1.0 --seed 42 "$@"

        run_step "${NAME}_gait" \
            conda run --no-capture-output -n piano python -u \
                scripts/stage_b_generator/round26_gait_diag.py \
                --config "${CFG_LOCAL}" --ckpt "${CKPT}" \
                --selection-json "${SELECTION_JSON}" \
                --output-dir "${OUT_DIR}" \
                --bucket train --cfg-scale 1.0 --seed 42 "$@"

        run_step "${NAME}_body_action" \
            conda run --no-capture-output -n piano python -u \
                scripts/stage_b_generator/round28_body_action_diag.py \
                --config "${CFG_LOCAL}" --ckpt "${CKPT}" \
                --selection-json "${SELECTION_JSON}" \
                --output-dir "${OUT_DIR}" \
                --bucket train --cfg-scale 1.0 --seed 42 "$@"
    }

    # Balanced same-subset baselines (run once per pack).
    run_diagnostics "baseline_v27_final" \
        "${V27_CFG_LOCAL}" "${V27_CKPT}" \
        "analyses/round28_baseline_v27_diag_final" \
        "${EVAL_SUBSET_BALANCED}"
    run_diagnostics "gt_reference" \
        "${V27_CFG_LOCAL}" "${V27_CKPT}" \
        "analyses/round28_gt_reference_diag" \
        "${EVAL_SUBSET_BALANCED}" \
        --use-gt-as-pred
    if _any_body_subset_variant; then
        run_diagnostics "baseline_v27_body_final" \
            "${V27_CFG_LOCAL}" "${V27_CKPT}" \
            "analyses/round28_baseline_v27_body_diag_final" \
            "${EVAL_SUBSET_BODY}"
        run_diagnostics "gt_reference_body" \
            "${V27_CFG_LOCAL}" "${V27_CKPT}" \
            "analyses/round28_gt_reference_body_diag" \
            "${EVAL_SUBSET_BODY}" \
            --use-gt-as-pred
    fi

    for V in "${VARIANTS[@]}"; do
        CFG_LOCAL="$(_cfg_local "$V")"
        RUN_DIR="$(_run_dir "$V")"

        for TAG in best_val final; do
            CKPT="${RUN_DIR}/${TAG}.pt"
            if [[ ! -f "${CKPT}" ]]; then
                echo "[skip] ${V}_${TAG} missing ${CKPT}"
                continue
            fi
            OUT_DIR="analyses/round28_${V}_diag_${TAG}"
            SEL="${EVAL_SUBSET_BALANCED}"
            if _uses_body_subset "${V}"; then
                SEL="${EVAL_SUBSET_BODY}"
            fi
            run_diagnostics "${V}_${TAG}" "${CFG_LOCAL}" "${CKPT}" "${OUT_DIR}" "${SEL}"
        done
    done
fi

# ============================================================
# PACK
# ============================================================
echo
echo "================================================================"
echo "[$(date '+%F %T')] PACK results"
echo "================================================================"
run_step "pack" \
    bash scripts/stage_b_generator/round28_pack_results.sh

echo
echo "================================================================"
echo "Round-28 train+eval complete."
echo "  variants run:       ${VARIANTS[*]}"
echo "  Diagnostic outputs: analyses/round28_*_diag_*"
echo "  Logs:               ${LOG_DIR}/*.log"
echo "  Tarball:            round28_results_*.tar.gz at project root"
echo "================================================================"
