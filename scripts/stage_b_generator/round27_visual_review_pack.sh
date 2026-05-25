#!/usr/bin/env bash
# Round-27 Tier-0 visual review — pack outputs into a single tarball.
#
# Includes (~200-500 MB depending on N_CLIPS):
#   - analyses/round27_visual_review/{v27_baseline,t0a3_contact_winner,t0b1_gait_winner}/
#       clip*_{gt,pred}.mp4 + summary.md
#   - analyses/round27_tier0_eval_selection_balanced.json   (selection used)
#   - analyses/round27_tier0_train_indices_48_balanced.json (clip set context)
#   - runs/round27_visual_review/*.log                       (render logs)
#
# Usage:
#   bash scripts/stage_b_generator/round27_visual_review_pack.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

STAMP="$(date +%Y-%m-%d_%H%M)"
OUT="round27_visual_review_${STAMP}.tar.gz"

ITEMS=()

# Video output dirs.
for dir in \
    analyses/round27_visual_review/v27_baseline \
    analyses/round27_visual_review/t0a3_contact_winner \
    analyses/round27_visual_review/t0b1_gait_winner ; do
    if [[ -d "${dir}" ]]; then
        for f in "${dir}"/*.mp4 "${dir}"/summary.md ; do
            [[ -e "$f" ]] && ITEMS+=("$f")
        done
    else
        echo "  [skip] missing render dir: ${dir}"
    fi
done

# Selection + clip-set context (small; bundle for self-containment).
for f in \
    analyses/round27_tier0_eval_selection_balanced.json \
    analyses/round27_tier0_train_indices_48_balanced.json ; do
    [[ -e "$f" ]] && ITEMS+=("$f")
done

# Stage logs.
for f in runs/round27_visual_review/*.log ; do
    [[ -e "$f" ]] && ITEMS+=("$f")
done

if [[ ${#ITEMS[@]} -eq 0 ]]; then
    echo "ERROR: no Round-27 visual review outputs found."
    exit 1
fi

echo "Packing ${#ITEMS[@]} items into ${OUT}:"
for x in "${ITEMS[@]}"; do
    echo "  + ${x}"
done

tar -czf "${OUT}" "${ITEMS[@]}"
ABS_OUT="$(realpath "${OUT}")"
SIZE="$(du -h "${OUT}" | awk '{print $1}')"

echo
echo "================================================================"
echo "Wrote: ${ABS_OUT}  (${SIZE})"
echo "================================================================"
echo "Transfer to local:"
echo "  scp <server>:${ABS_OUT} ."
