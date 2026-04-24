#!/usr/bin/env bash
# Distance-threshold sweep for contact extraction.
#
# For every InterAct subset, computes raw joint-to-mesh distances once
# and then applies a threshold grid to report per-body-part frame_rate +
# seq_reached. Outputs land under runs/threshold_sweep/<ts>/<subset>/.
#
# The collect phase is ~similar cost to pseudo-label extraction (one
# mesh distance query per frame per body part). The analysis phase is
# cheap (< 1 min).
#
# Usage:
#   bash scripts/stage1_pseudo_labels/threshold_sweep.sh
#   bash scripts/stage1_pseudo_labels/threshold_sweep.sh chairs           # one subset
#   OUTPUT_ROOT=custom/path bash scripts/stage1_pseudo_labels/threshold_sweep.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

PIANO_ROOT="${PIANO_ROOT:-/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano}"
INTERACT_ROOT="${INTERACT_ROOT:-/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/InterAct}"
SUBSETS=("chairs" "imhd" "neuraldome" "omomo_correct_v2")

if [[ $# -gt 0 ]]; then
    SUBSETS=("$@")
fi

TS=$(date +%Y-%m-%d_%H%M%S)
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/threshold_sweep/$TS}"
mkdir -p "$OUTPUT_ROOT"

for subset in "${SUBSETS[@]}"; do
    echo ""
    echo "=========================================================="
    echo "Threshold sweep: $subset"
    echo "=========================================================="
    data_dir="$PIANO_ROOT/$subset"
    mesh_dir="$INTERACT_ROOT/$subset/objects"
    out_dir="$OUTPUT_ROOT/$subset"

    if [[ ! -d "$data_dir" ]]; then
        echo "  [skip] $data_dir not found"
        continue
    fi

    python -m piano.checks.threshold_sweep \
        --data-dir "$data_dir" \
        --mesh-dir "$mesh_dir" \
        --output-dir "$out_dir" \
        --subset "$subset" \
        --mesh-suffixes "_face1000" "_simplified" ""
done

echo ""
echo "Outputs under: $OUTPUT_ROOT"
