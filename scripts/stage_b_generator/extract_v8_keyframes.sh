#!/usr/bin/env bash
# v8 offline keyframe extraction across all 4 InterAct subsets.
# Output: <data_dir>/keyframes/v8_default/<seq_id>.npz per clip
#   - indices: (K,) int32 frame indices, K in [5, 12]
#   - targets: (K, 6, 3) float32 world XYZ for (root, L_hand, R_hand,
#     L_foot, R_foot, head)
#   - num_keyframes: scalar K
#
# ~15 min CPU total (single-threaded, ~8K clips). Resume support
# (skips existing .npz). Run alongside v19 extraction (independent).
set -euo pipefail

cd "$(dirname "$0")/../.."

PIANO_ROOT="${PIANO_ROOT:-E:/Project/Datasets/InterAct/piano_official_process_4}"
PYTHON="${PYTHON:-C:/Users/cwq29/miniforge3/envs/piano/python.exe}"
LABEL_SUBDIR="pseudo_labels/v18_h10_f05_pelvis20_official_semantic_marker"
OUTPUT_SUBDIR="keyframes/v8_default"
SUBSETS=("chairs" "imhd" "neuraldome" "omomo_correct_v2")

if [[ $# -gt 0 ]]; then
    SUBSETS=("$@")
fi

for subset in "${SUBSETS[@]}"; do
    echo "=========================================================="
    echo "Extracting v8 keyframes for InterAct subset: $subset"
    echo "=========================================================="
    data_dir="$PIANO_ROOT/$subset"

    if [[ ! -d "$data_dir" ]]; then
        echo "  [skip] $data_dir not found"
        continue
    fi

    "$PYTHON" -m piano.data.keyframe_extraction \
        --data-dir "$data_dir" \
        --pseudo-label-subdir "$LABEL_SUBDIR" \
        --output-subdir "$OUTPUT_SUBDIR"
done

echo ""
echo "All v8 keyframe extraction done."
