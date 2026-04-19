#!/usr/bin/env bash
# Extract pseudo interaction labels for all InterAct subsets.
# Runs the generic pseudo_labels.run_all once per subset, pointing at that
# subset's preprocessed PIANO root and its original object mesh directory.
#
# Default paths (server):
#   piano-data (per subset):  /media/.../InterAct/piano/<subset>
#   mesh source (per subset): /media/.../InterAct/InterAct/<subset>/objects
#
# Usage:
#   bash scripts/data/extract_pseudo_labels_interact.sh              # all 4 subsets
#   bash scripts/data/extract_pseudo_labels_interact.sh chairs       # one subset only
set -euo pipefail

cd "$(dirname "$0")/../.."

PIANO_ROOT="/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano"
INTERACT_ROOT="/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/InterAct"
SUBSETS=("chairs" "imhd" "neuraldome" "omomo_correct_v2")

# Allow positional args to select specific subsets
if [[ $# -gt 0 ]]; then
    SUBSETS=("$@")
fi

for subset in "${SUBSETS[@]}"; do
    echo "=========================================================="
    echo "Extracting pseudo-labels for InterAct subset: $subset"
    echo "=========================================================="
    data_dir="$PIANO_ROOT/$subset"
    mesh_dir="$INTERACT_ROOT/$subset/objects"
    output_dir="$data_dir/pseudo_labels"

    if [[ ! -d "$data_dir" ]]; then
        echo "  [skip] $data_dir not found (run preprocess_interact first)"
        continue
    fi

    # Prefer simplified mesh variants for speed/memory. run_all will try
    # each suffix in order until it finds a file.
    python -m piano.data.pseudo_labels.run_all \
        --data-dir "$data_dir" \
        --mesh-dir "$mesh_dir" \
        --output-dir "$output_dir" \
        --mesh-suffixes "_face1000" "_simplified" ""
done

echo ""
echo "All subsets done."
