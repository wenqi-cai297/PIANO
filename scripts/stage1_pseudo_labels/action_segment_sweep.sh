#!/usr/bin/env bash
# Action-segment contact sweep across all 4 subsets.
# Reuses cached distances.npz from a previous piano-threshold-sweep run.
#
# Required positional arg: the sweep timestamp, e.g. 2026-04-20_193818.
# Outputs are written next to the cached distances, so they stay
# co-located with the plain sweep results.
#
# Usage:
#   bash scripts/stage1_pseudo_labels/action_segment_sweep.sh 2026-04-20_193818
#   bash scripts/stage1_pseudo_labels/action_segment_sweep.sh 2026-04-20_193818 chairs imhd
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <sweep_timestamp> [subset ...]"
    echo "  where runs/threshold_sweep/<sweep_timestamp>/<subset>/distances.npz exists"
    exit 1
fi

SWEEP_TS="$1"; shift
SWEEP_DIR="runs/threshold_sweep/$SWEEP_TS"

INTERACT_ROOT="${INTERACT_ROOT:-/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/InterAct}"
ALL_SUBSETS=("chairs" "imhd" "neuraldome" "omomo_correct_v2")

if [[ $# -gt 0 ]]; then
    SUBSETS=("$@")
else
    SUBSETS=("${ALL_SUBSETS[@]}")
fi

for subset in "${SUBSETS[@]}"; do
    distances_npz="$SWEEP_DIR/$subset/distances.npz"
    interact_subset="$INTERACT_ROOT/$subset"
    if [[ ! -f "$distances_npz" ]]; then
        echo "[skip] $distances_npz not found"
        continue
    fi
    if [[ ! -d "$interact_subset/sequences_canonical" ]]; then
        echo "[skip] $interact_subset/sequences_canonical not found (cannot parse text.txt)"
        continue
    fi
    echo ""
    echo "=========================================================="
    echo "Action-segment sweep: $subset"
    echo "=========================================================="
    python -m piano.checks.action_segment_sweep \
        --distances-npz "$distances_npz" \
        --interact-dir "$interact_subset" \
        --output-dir "$SWEEP_DIR/$subset" \
        --subset "$subset"
done

echo ""
echo "Outputs under $SWEEP_DIR/<subset>/action_segment_analysis.{json,md}"
