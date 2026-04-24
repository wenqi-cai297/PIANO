#!/usr/bin/env bash
# Compute rich pseudo-label quality stats on already-extracted labels.
# Writes stats.json + stats.md next to each subset's existing
# pseudo_labels/summary.json (summary.json itself is left untouched).
#
# Usage:
#   bash scripts/stage1_pseudo_labels/pseudo_label_stats.sh                    # all 4 subsets
#   bash scripts/stage1_pseudo_labels/pseudo_label_stats.sh chairs imhd        # selected
#   PIANO_ROOT=/path/to/piano bash scripts/stage1_pseudo_labels/pseudo_label_stats.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

PIANO_ROOT="${PIANO_ROOT:-/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano}"
SUBSETS=("chairs" "imhd" "neuraldome" "omomo_correct_v2")

if [[ $# -gt 0 ]]; then
    SUBSETS=("$@")
fi

for subset in "${SUBSETS[@]}"; do
    data_dir="$PIANO_ROOT/$subset"
    if [[ ! -d "$data_dir/pseudo_labels" ]]; then
        echo "[skip] $data_dir/pseudo_labels not found"
        continue
    fi
    echo ""
    echo "=========================================================="
    echo "Stats for subset: $subset"
    echo "=========================================================="
    python -m piano.checks.pseudo_label_stats \
        --data-dir "$data_dir" \
        --subset "$subset"
done

echo ""
echo "All subsets done. stats.json / stats.md live next to each subset's summary.json."
