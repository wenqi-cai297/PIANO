#!/usr/bin/env bash
# Round-27 Tier-0A {hand, foot, full} oracle interaction-hint diagnostic.
# Per piano_stage2_full_architecture_roadmap.md §6 / §8.2-§8.4 / §10.2.
#
# What this runs (per Tier-0A variant t0a1/t0a2/t0a3):
#   1. TRAIN  48-clip overfit, 300 epochs, fine-tune from v27 final.pt.
#             (~15-20 min on dual-A6000; ~30-40 min single-GPU.)
#   2. EVAL   sustained-contact + gait diagnostics on the same 48 clips
#             (use the train bucket since this is an overfit diagnostic
#             — roadmap §10.1 success thresholds are stated for the
#             overfit set itself).
#   3. PACK   tarball results + logs (no .pt ckpts, ~50 MB total).
#
# Wall-clock budget (dual-A6000):
#   PREP        ~1 min   (translate Windows-paths YAML to Linux)
#   TRAIN ×3    ~60 min  (3 × ~20 min sequentially; can be parallelised
#                         if you split GPU 0/1, but simpler to keep one
#                         DDP run at a time so each gets the full machine)
#   EVAL ×3     ~15 min  (sustained-contact + gait per ckpt; tiny)
#   PACK        ~30 sec
#   TOTAL       ~75 min
#
# Prerequisites on the Linux server:
#   1. git pull (must include the Round-27 Commit 1 + 2 commits)
#   2. v27 ckpt at:
#         runs/training/stageB_anchordiff_v27_stage2_anchoraware_FULL_DATA/final.pt
#   3. 48-clip balanced train indices file:
#         analyses/round27_tier0_train_indices_48_balanced.json
#   4. Stage-1 oracle Coarse-v1 cache:
#         cache/stage1_coarse_v1_full
#
# Usage:
#   bash scripts/stage_b_generator/run_round27_t0a_train.sh                         # all 6 variants
#   bash scripts/stage_b_generator/run_round27_t0a_train.sh t0a1                    # one variant only
#   bash scripts/stage_b_generator/run_round27_t0a_train.sh t0a1 t0a2               # subset
#
# Skip-stage:
#   ROUND27_RESUME_FROM=eval bash <this>     # skip PREP + TRAIN
#   ROUND27_RESUME_FROM=pack bash <this>     # only pack
#
# Single-GPU fallback:
#   ROUND27_SINGLE_GPU=1 bash <this>
#
set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="runs/round27_t0a_train"
mkdir -p "${LOG_DIR}"

V27_CKPT="runs/training/stageB_anchordiff_v27_stage2_anchoraware_FULL_DATA/final.pt"
R23_CKPT="runs/training/stageB_anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA/final.pt"
V27_CFG_LOCAL="configs/training/anchordiff_v27_stage2_anchoraware_FULL_DATA_local.yaml"
R23_CFG_LOCAL="configs/training/anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA_local.yaml"
TRAIN_INDICES="${ROUND27_TRAIN_INDICES:-analyses/round27_tier0_train_indices_48_balanced.json}"
MAX_CANDIDATES_PER_SUBSET="${ROUND27_MAX_CANDIDATES_PER_SUBSET:-600}"

SINGLE_GPU="${ROUND27_SINGLE_GPU:-0}"
RESUME_FROM="${ROUND27_RESUME_FROM:-}"

# Default: run all six Tier-0 variants. CLI args narrow it down.
# Variants: t0a1 (hand) t0a2 (foot) t0a3 (full) t0b1 (temporal-from-v27)
#           t0b2 (temporal-from-R23) t0ab (full+temporal upper bound)
if [[ $# -gt 0 ]]; then
    VARIANTS=("$@")
else
    VARIANTS=(t0a1 t0a2 t0a3 t0b1 t0b2 t0ab)
fi

_should_skip() {
    local stages=(prep train eval pack)
    [[ -z "${RESUME_FROM}" ]] && return 1
    local target_idx=-1 current_idx=-1 i
    for ((i=0; i<${#stages[@]}; i++)); do
        [[ "${stages[i]}" == "$1" ]] && current_idx=$i
        [[ "${stages[i]}" == "${RESUME_FROM}" ]] && target_idx=$i
    done
    [[ $target_idx -lt 0 ]] && { echo "WARN: unknown ROUND27_RESUME_FROM=${RESUME_FROM}"; return 1; }
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

# Per-variant lookup table.
_variant_label() {
    case "$1" in
        t0a1) echo "hand-only oracle hint (D=8)" ;;
        t0a2) echo "foot-only oracle hint (D=5)" ;;
        t0a3) echo "full oracle hint (D=13)" ;;
        t0b1) echo "temporal losses only, from v27" ;;
        t0b2) echo "temporal losses only, from R23" ;;
        t0ab) echo "full hint + temporal losses (upper bound)" ;;
        *) echo "unknown" ;;
    esac
}

_variant_config_local() {
    case "$1" in
        t0a1) echo "configs/training/anchordiff_t0a1_hand_oracle_hint_48clip_local.yaml" ;;
        t0a2) echo "configs/training/anchordiff_t0a2_foot_oracle_hint_48clip_local.yaml" ;;
        t0a3) echo "configs/training/anchordiff_t0a3_full_oracle_hint_48clip_local.yaml" ;;
        t0b1) echo "configs/training/anchordiff_t0b1_temporal_losses_48clip_from_v27_local.yaml" ;;
        t0b2) echo "configs/training/anchordiff_t0b2_temporal_losses_48clip_from_r23_local.yaml" ;;
        t0ab) echo "configs/training/anchordiff_t0ab_full_oracle_hint_temporal_losses_48clip_local.yaml" ;;
        *) echo ""; return 1 ;;
    esac
}

_variant_run_dir() {
    case "$1" in
        t0a1) echo "runs/training/stageB_anchordiff_t0a1_hand_oracle_hint_48clip" ;;
        t0a2) echo "runs/training/stageB_anchordiff_t0a2_foot_oracle_hint_48clip" ;;
        t0a3) echo "runs/training/stageB_anchordiff_t0a3_full_oracle_hint_48clip" ;;
        t0b1) echo "runs/training/stageB_anchordiff_t0b1_temporal_losses_48clip_from_v27" ;;
        t0b2) echo "runs/training/stageB_anchordiff_t0b2_temporal_losses_48clip_from_r23" ;;
        t0ab) echo "runs/training/stageB_anchordiff_t0ab_full_oracle_hint_temporal_losses_48clip" ;;
        *) echo ""; return 1 ;;
    esac
}

# ============================================================
# PREP: translate Tier-0A YAMLs to Linux paths
# ============================================================
if _should_skip prep; then
    echo "[SKIP] PREP (ROUND27_RESUME_FROM=${RESUME_FROM})"
else
    run_step "prep_make_local_configs" \
        bash scripts/stage_b_generator/run_round27_make_local_configs.sh
fi

# ---------- build train indices file if missing ----------
# analyses/ is gitignored, so the 48-clip indices file may be absent on
# the server. Re-generate it with the balanced Round-27 builder; the old
# Round-25 category builder produced all-chair clips.
if [[ ! -f "${TRAIN_INDICES}" ]]; then
    echo "[$(date '+%F %T')] Building train indices: ${TRAIN_INDICES}"
    if [[ ! -f "${V27_CFG_LOCAL}" ]]; then
        # The builder reads the dataset roots from a *local* config so paths
        # are resolved on this machine. Use the v27 local config (PREP just
        # generated it from the Tier-0A PREP step's sed call — round27 prep
        # is the t0a* configs; we ALSO need v27_local for this builder).
        # Trigger v27 local-config generation explicitly here.
        run_step "prep_v27_local_for_indices_builder" \
            bash scripts/stage_b_generator/run_round26_make_local_configs.sh
    fi
    run_step "build_train_indices" \
        conda run --no-capture-output -n piano python -u \
            scripts/stage_b_generator/round27_build_tier0_train_indices.py \
            --config "${V27_CFG_LOCAL}" \
            --n-clips 48 \
            --output "${TRAIN_INDICES}" \
            --max-candidates-per-subset "${MAX_CANDIDATES_PER_SUBSET}"
fi

# ---------- preflight ----------
for V in "${VARIANTS[@]}"; do
    CFG_LOCAL="$(_variant_config_local "$V")"
    [[ -f "${CFG_LOCAL}" ]] || { echo "ERROR: missing config: ${CFG_LOCAL}"; exit 1; }
done

# Init ckpts: every variant except t0b2 starts from v27; t0b2 starts from R23.
_needs_v27=0; _needs_r23=0
for V in "${VARIANTS[@]}"; do
    if [[ "${V}" == "t0b2" ]]; then _needs_r23=1; else _needs_v27=1; fi
done
if [[ $_needs_v27 -eq 1 && ! -f "${V27_CKPT}" ]]; then
    echo "ERROR: missing v27 ckpt: ${V27_CKPT}"; exit 1
fi
if [[ $_needs_r23 -eq 1 && ! -f "${R23_CKPT}" ]]; then
    echo "ERROR: missing R23 ckpt (needed by t0b2): ${R23_CKPT}"; exit 1
fi
[[ -f "${V27_CKPT}" ]] || { echo "ERROR: missing v27 baseline ckpt: ${V27_CKPT}"; exit 1; }
[[ -f "${R23_CKPT}" ]] || { echo "ERROR: missing R23 baseline ckpt: ${R23_CKPT}"; exit 1; }
[[ -f "${TRAIN_INDICES}" ]] || { echo "ERROR: missing train indices: ${TRAIN_INDICES}"; exit 1; }
[[ -d "cache/stage1_coarse_v1_full" ]] || { echo "ERROR: missing Stage-1 cache"; exit 1; }

# ============================================================
# TRAIN: sequential over variants (each consumes the full machine for DDP)
# ============================================================
if _should_skip train; then
    echo "[SKIP] TRAIN (ROUND27_RESUME_FROM=${RESUME_FROM})"
else
    for V in "${VARIANTS[@]}"; do
        CFG_LOCAL="$(_variant_config_local "$V")"
        LABEL="$(_variant_label "$V")"

        echo "================================================================"
        echo "[$(date '+%F %T')] TRAIN ${V} (${LABEL})"
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
# EVAL: sustained-contact + gait on best_val.pt and final.pt per variant
# ============================================================
# Both diagnostics use --bucket val by default; we override to --bucket train
# and pass --selection-json mapping (subset, seq_id) of the 48 clips.
# Since the indices file is train-indices (not val), we re-generate a
# selection-json with the (subset, seq_id) of the 48 train clips so the
# diag scripts can resolve them via their standard selection-json reader.
EVAL_SUBSET="${EVAL_SUBSET:-analyses/round27_tier0_eval_selection_balanced.json}"

if _should_skip eval; then
    echo "[SKIP] EVAL (ROUND27_RESUME_FROM=${RESUME_FROM})"
else
    # Build the (subset, seq_id) selection JSON from the indices file
    # if it doesn't exist. The two diag scripts expect the same schema
    # as analyses/round25_multimodal_eval_subset.json.
    if [[ ! -f "${EVAL_SUBSET}" ]]; then
        echo "[$(date '+%F %T')] Building eval selection JSON from train indices..."
        conda run --no-capture-output -n piano python -c "
import json
src = json.load(open('${TRAIN_INDICES}', encoding='utf-8'))
clips = src['clips']
out = {
    'description': 'Round-27 Tier-0A train-bucket overfit selection (auto-generated from ${TRAIN_INDICES}).',
    'selection_source': '${TRAIN_INDICES}',
    'bucket': 'train',
    'n_clips': len(clips),
    'selected': [
        {'subset': c['subset'], 'seq_id': c['seq_id'],
         'mode_category': c.get('mode_category', 'unknown'),
         'text': '', 'confidence': 1.0, 'n_known_valid_modes': 1}
        for c in clips
    ],
}
json.dump(out, open('${EVAL_SUBSET}', 'w', encoding='utf-8'), indent=2)
print(f'wrote {len(clips)} clips to ${EVAL_SUBSET}')
"
    fi

    if [[ ! -f "${V27_CFG_LOCAL}" ]]; then
        run_step "prep_v27_local_for_eval_baseline" \
            bash scripts/stage_b_generator/run_round26_make_local_configs.sh
    fi
    if [[ ! -f "${R23_CFG_LOCAL}" ]]; then
        run_step "prep_r23_local_for_eval_baseline" \
            bash scripts/stage_b_generator/run_round23_make_local_configs.sh
    fi

    run_diagnostics() {
        local NAME="$1"; shift
        local CFG_LOCAL="$1"; shift
        local CKPT="$1"; shift
        local OUT_DIR="$1"; shift
        mkdir -p "${OUT_DIR}"

        run_step "${NAME}_sustained" \
            conda run --no-capture-output -n piano python -u \
                scripts/stage_b_generator/round26_sustained_contact_diag.py \
                --config "${CFG_LOCAL}" --ckpt "${CKPT}" \
                --selection-json "${EVAL_SUBSET}" \
                --output-dir "${OUT_DIR}" \
                --bucket train --cfg-scale 1.0 --seed 42 "$@"

        run_step "${NAME}_gait" \
            conda run --no-capture-output -n piano python -u \
                scripts/stage_b_generator/round26_gait_diag.py \
                --config "${CFG_LOCAL}" --ckpt "${CKPT}" \
                --selection-json "${EVAL_SUBSET}" \
                --output-dir "${OUT_DIR}" \
                --bucket train --cfg-scale 1.0 --seed 42 "$@"
    }

    # Same-subset references: no-finetune baselines and GT-as-pred sanity.
    run_diagnostics "baseline_v27_final" \
        "${V27_CFG_LOCAL}" "${V27_CKPT}" \
        "analyses/round27_baseline_v27_diag_final"
    run_diagnostics "baseline_r23_final" \
        "${R23_CFG_LOCAL}" "${R23_CKPT}" \
        "analyses/round27_baseline_r23_diag_final"
    run_diagnostics "gt_reference" \
        "${V27_CFG_LOCAL}" "${V27_CKPT}" \
        "analyses/round27_gt_reference_diag" \
        --use-gt-as-pred

    for V in "${VARIANTS[@]}"; do
        CFG_LOCAL="$(_variant_config_local "$V")"
        RUN_DIR="$(_variant_run_dir "$V")"

        for TAG in best_val final; do
            CKPT="${RUN_DIR}/${TAG}.pt"
            if [[ ! -f "${CKPT}" ]]; then
                echo "[skip] ${V}_${TAG} missing ${CKPT}"
                continue
            fi

            OUT_DIR="analyses/round27_${V}_diag_${TAG}"
            run_diagnostics "${V}_${TAG}" "${CFG_LOCAL}" "${CKPT}" "${OUT_DIR}"
        done
    done
fi

# ============================================================
# PACK: tarball
# ============================================================
echo
echo "================================================================"
echo "[$(date '+%F %T')] PACK results"
echo "================================================================"
run_step "pack" \
    bash scripts/stage_b_generator/round27_t0a_pack_results.sh

echo
echo "================================================================"
echo "Round-27 Tier-0A train+eval complete."
echo "  variants run: ${VARIANTS[*]}"
echo "  Diagnostic outputs: analyses/round27_*_diag_*"
echo "  Logs:               ${LOG_DIR}/*.log"
echo "  Tarball:            round27_t0a_results_*.tar.gz at project root"
echo "================================================================"
