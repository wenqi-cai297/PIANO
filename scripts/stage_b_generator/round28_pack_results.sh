#!/usr/bin/env bash
# Round-28 train+diag: pack all analysis-relevant files into a single
# tarball for transfer to the local machine.
#
# Includes (~50-80 MB):
#   - analyses/round28_<variant>_diag_{best_val,final}/*: per-variant
#       sustained-contact + gait + body-action diag outputs
#   - analyses/round28_baseline_v27*_diag_final/*: v27 baseline
#   - analyses/round28_gt_reference*diag/*: GT-as-pred sanity
#   - analyses/round27_tier0_train_indices_48_balanced.json: balanced selection
#   - analyses/round27_tier0_eval_selection_balanced.json
#   - analyses/round28_body_action_{train_indices_48,eval_selection}.json
#   - analyses/round28_claude_code_stage2_oracle_interface_prompt.md
#   - runs/round28_train/*.log: stage logs
#   - runs/training/stageB_anchordiff_r28_*/metrics.jsonl + *.log + *.txt + *.yaml
#
# Excludes:
#   - *.pt checkpoints
#   - wandb/ subdirs
#
# Usage:
#   bash scripts/stage_b_generator/round28_pack_results.sh
#
# Output: round28_results_YYYY-MM-DD_HHMM.tar.gz at project root.

set -euo pipefail
cd "$(dirname "$0")/../.."

STAMP="$(date +%Y-%m-%d_%H%M)"
OUT="round28_results_${STAMP}.tar.gz"

ITEMS=()

VARIANTS=(
    r28_a0_input_add
    r28_a1_gated_input
    r28_a2_per_layer_adapter
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

# Per-variant diagnostic outputs.
for V in "${VARIANTS[@]}"; do
    for TAG in best_val final; do
        DIR="analyses/round28_${V}_diag_${TAG}"
        [[ -d "${DIR}" ]] && ITEMS+=("${DIR}")
    done
done
for DIR in \
    analyses/round28_baseline_v27_diag_final \
    analyses/round28_baseline_v27_body_diag_final \
    analyses/round28_gt_reference_diag \
    analyses/round28_gt_reference_body_diag
do
    [[ -d "${DIR}" ]] && ITEMS+=("${DIR}")
done

# Reference selections.
for f in \
    analyses/round27_tier0_train_indices_48_balanced.json \
    analyses/round27_tier0_eval_selection_balanced.json \
    analyses/round28_body_action_train_indices_48.json \
    analyses/round28_body_action_eval_selection.json \
    analyses/round28_claude_code_stage2_oracle_interface_prompt.md
do
    [[ -f "${f}" ]] && ITEMS+=("${f}")
done

# Stage logs.
[[ -d "runs/round28_train" ]] && ITEMS+=("runs/round28_train")

# Per-variant training metadata (NOT ckpts).
shopt -s nullglob
for V in "${VARIANTS[@]}"; do
    RUN_DIR="runs/training/stageB_anchordiff_${V}_48clip"
    [[ -d "${RUN_DIR}" ]] || continue
    for ext in jsonl log txt yaml; do
        for f in "${RUN_DIR}"/*."${ext}"; do
            [[ -f "${f}" ]] && ITEMS+=("${f}")
        done
    done
done
shopt -u nullglob

if [[ ${#ITEMS[@]} -eq 0 ]]; then
    echo "ERROR: no Round-28 outputs found to pack."
    exit 1
fi

echo "Packing ${#ITEMS[@]} items into ${OUT}:"
for x in "${ITEMS[@]}"; do
    echo "  + ${x}"
done

tar --exclude='*.pt' --exclude='wandb' -czf "${OUT}" "${ITEMS[@]}"
ABS_OUT="$(realpath "${OUT}")"
SIZE="$(du -h "${OUT}" | awk '{print $1}')"

echo
echo "================================================================"
echo "Wrote: ${ABS_OUT}  (${SIZE})"
echo "================================================================"
echo "Transfer to local:"
echo "  scp <server>:${ABS_OUT} ."
