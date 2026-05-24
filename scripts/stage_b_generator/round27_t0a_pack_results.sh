#!/usr/bin/env bash
# Round-27 Tier-0A train+diag — pack all analysis-relevant files into a
# single tarball for transfer to the local machine.
#
# Includes (small, ~50 MB total):
#   - analyses/round27_t0a{1,2,3}_diag_{best_val,final}/*           — per-variant sustained-contact + gait diag outputs (json, md, optional npz / png)
#   - analyses/round27_tier0_train_indices_48.json                 — train subset definition (already in repo, but include for self-containedness)
#   - analyses/round27_tier0_eval_selection.json                   — auto-generated eval selection
#   - runs/round27_t0a_train/*.log                                  — stage logs
#   - runs/training/stageB_anchordiff_t0a*/metrics.jsonl + *.log + *.txt + *.yaml
#   - piano_stage2_full_architecture_roadmap.md                    — primary direction
#   - analyses/2026-05-25_round26_closure_roadmap_adopted_as_next_phase.md
#
# Excludes (large, server-only):
#   - *.pt checkpoints
#   - wandb/ subdirs
#
# Usage:
#   bash scripts/stage_b_generator/round27_t0a_pack_results.sh
#
# Output: round27_t0a_results_YYYY-MM-DD_HHMM.tar.gz at project root.

set -euo pipefail
cd "$(dirname "$0")/../.."

STAMP="$(date +%Y-%m-%d_%H%M)"
OUT="round27_t0a_results_${STAMP}.tar.gz"

ITEMS=()

# Per-variant diagnostic outputs (all 6 Tier-0 variants).
for V in t0a1 t0a2 t0a3 t0b1 t0b2 t0ab; do
    for TAG in best_val final; do
        DIR="analyses/round27_${V}_diag_${TAG}"
        [[ -d "${DIR}" ]] && ITEMS+=("${DIR}")
    done
done

# Reference selections + train indices.
for f in \
    analyses/round27_tier0_train_indices_48.json \
    analyses/round27_tier0_eval_selection.json
do
    [[ -f "${f}" ]] && ITEMS+=("${f}")
done

# Roadmap + closure note (small; ensures the tarball is self-explanatory).
for f in \
    piano_stage2_full_architecture_roadmap.md \
    analyses/2026-05-25_round26_closure_roadmap_adopted_as_next_phase.md
do
    [[ -f "${f}" ]] && ITEMS+=("${f}")
done

# Stage logs.
[[ -d "runs/round27_t0a_train" ]] && ITEMS+=("runs/round27_t0a_train")

# Per-variant training metadata (NOT ckpts).
shopt -s nullglob
declare -A RUN_DIR_BY_VARIANT=(
    [t0a1]="runs/training/stageB_anchordiff_t0a1_hand_oracle_hint_48clip"
    [t0a2]="runs/training/stageB_anchordiff_t0a2_foot_oracle_hint_48clip"
    [t0a3]="runs/training/stageB_anchordiff_t0a3_full_oracle_hint_48clip"
    [t0b1]="runs/training/stageB_anchordiff_t0b1_temporal_losses_48clip_from_v27"
    [t0b2]="runs/training/stageB_anchordiff_t0b2_temporal_losses_48clip_from_r23"
    [t0ab]="runs/training/stageB_anchordiff_t0ab_full_oracle_hint_temporal_losses_48clip"
)
for V in t0a1 t0a2 t0a3 t0b1 t0b2 t0ab; do
    RUN_DIR="${RUN_DIR_BY_VARIANT[$V]}"
    [[ -d "${RUN_DIR}" ]] || continue
    # Use --append style: include only non-ckpt files via tar --exclude.
    for ext in jsonl log txt yaml; do
        for f in "${RUN_DIR}"/*."${ext}"; do
            [[ -f "${f}" ]] && ITEMS+=("${f}")
        done
    done
done
shopt -u nullglob

if [[ ${#ITEMS[@]} -eq 0 ]]; then
    echo "ERROR: no Round-27 Tier-0A outputs found to pack."
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
echo "  (or rsync ${ABS_OUT} ...)"
