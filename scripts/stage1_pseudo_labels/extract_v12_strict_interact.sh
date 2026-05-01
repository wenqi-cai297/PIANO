#!/usr/bin/env bash
# Extract v12 STRICT pseudo-labels for all 4 InterAct subsets.
# Outputs to <subset>/pseudo_labels/v12_strict/ (parallel to v11's
# pseudo_labels/), so existing v11 paths are untouched.
#
# Definition: analyses/2026-05-03_pseudo_label_v12_strict_design.md (r3)
#   - Loose-distance threshold (hand 25 cm, foot 15, pelvis 30) gates
#     the contact decision; engagement (kinematic OR static) is the
#     primary signal.
#   - kin_local_sigma 0.06 m allows wrap-grip wrist articulation.
#   - max_segment_drift 0.10 m allows wrist articulation in object-local.
#   - duration filter ≥ 5 frames + median filter 7 frames + drift filter.
#
# Usage:
#   bash scripts/stage1_pseudo_labels/extract_v12_strict_interact.sh
#                                                      # all 4 subsets
#   bash scripts/stage1_pseudo_labels/extract_v12_strict_interact.sh chairs
#                                                      # one subset only
set -euo pipefail

cd "$(dirname "$0")/../.."

PIANO_ROOT="${PIANO_ROOT:-/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano}"
INTERACT_ROOT="${INTERACT_ROOT:-/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/InterAct}"
LABEL_VERSION="${LABEL_VERSION:-v12_strict}"
SUBSETS=("chairs" "imhd" "neuraldome" "omomo_correct_v2")

if [[ $# -gt 0 ]]; then
    SUBSETS=("$@")
fi

t_global_start=$(date +%s)
for subset in "${SUBSETS[@]}"; do
    echo "=========================================================="
    echo "[v12_strict] Extracting pseudo-labels: $subset"
    echo "=========================================================="
    data_dir="$PIANO_ROOT/$subset"
    mesh_dir="$INTERACT_ROOT/$subset/objects"
    output_dir="$data_dir/pseudo_labels/$LABEL_VERSION"

    if [[ ! -d "$data_dir" ]]; then
        echo "  [skip] $data_dir not found (run preprocess_interact first)"
        continue
    fi

    mkdir -p "$output_dir"

    python -m piano.data.pseudo_labels.run_all \
        --data-dir "$data_dir" \
        --mesh-dir "$mesh_dir" \
        --output-dir "$output_dir" \
        --contact-version v12_strict \
        --mesh-suffixes "_face1000" "_simplified" ""
done

elapsed=$(( $(date +%s) - t_global_start ))
echo ""
echo "All subsets done in ${elapsed}s."
echo ""
echo "Next step: compare v12 vs v11 frame fractions:"
echo "  python scripts/stage1_pseudo_labels/compare_v11_v12_strict.py \\"
echo "      --piano-root $PIANO_ROOT"
