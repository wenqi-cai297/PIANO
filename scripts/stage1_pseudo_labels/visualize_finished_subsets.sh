#!/usr/bin/env bash
# Visualize pseudo-labels for all InterAct subsets that have finished
# extraction (detected by presence of pseudo_labels/summary.json).
#
# For each finished subset:
#   1. Pick a handful of "representative" sequences by matching text
#      keywords (sit, lift, move, pick, etc.)
#   2. Render MP4 videos with contact/phase/support overlay
#   3. Write summary.json
#
# Outputs land under: runs/visualizations/<timestamp>_pseudo_labels_<subset>/
#
# Usage:
#   bash scripts/stage1_pseudo_labels/visualize_finished_subsets.sh
#   bash scripts/stage1_pseudo_labels/visualize_finished_subsets.sh chairs imhd   # subset selection
set -euo pipefail

cd "$(dirname "$0")/../.."

PIANO_ROOT="/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano"
ALL_SUBSETS=("chairs" "imhd" "neuraldome" "omomo_correct_v2")

if [[ $# -gt 0 ]]; then
    SUBSETS=("$@")
else
    SUBSETS=("${ALL_SUBSETS[@]}")
fi

for subset in "${SUBSETS[@]}"; do
    data_dir="$PIANO_ROOT/$subset"
    summary="$data_dir/pseudo_labels/summary.json"

    if [[ ! -f "$summary" ]]; then
        echo "=== $subset: NOT finished yet, skipping"
        continue
    fi

    echo ""
    echo "=========================================================="
    echo "=== $subset: picking representative sequences"
    echo "=========================================================="

    ts=$(date +%Y-%m-%d_%H%M%S)
    pick_dir="runs/checks/sample_seq_by_keyword/${ts}_${subset}"

    python -m piano.checks.sample_seq_by_keyword \
        --data-dir "$data_dir" \
        --per-keyword 1 \
        --output-dir "$pick_dir"

    # Read seq_ids.txt (one per line) into an array
    mapfile -t SEQ_IDS < "$pick_dir/seq_ids.txt"

    if [[ ${#SEQ_IDS[@]} -eq 0 ]]; then
        echo "  no matching sequences found for $subset"
        continue
    fi

    echo ""
    echo "=== $subset: rendering ${#SEQ_IDS[@]} videos"
    viz_dir="runs/visualizations/${ts}_pseudo_labels_${subset}"
    python -m piano.inference.visualize_pseudo_labels \
        --data-dir "$data_dir" \
        --seq-ids "${SEQ_IDS[@]}" \
        --output-dir "$viz_dir"
done

echo ""
echo "All finished subsets visualized."
