#!/usr/bin/env bash
# Render pseudo-labels overlaid on skeleton videos.
# Run AFTER preprocess_interact + extract_pseudo_labels_interact.
#
# Usage:
#   # 4 random samples from a subset (default: omomo_correct_v2)
#   bash scripts/stage1_pseudo_labels/visualize_pseudo_labels.sh
#
#   # Specific subset
#   bash scripts/stage1_pseudo_labels/visualize_pseudo_labels.sh \
#       --data-dir /media/.../InterAct/piano/chairs
#
#   # Specific sequences
#   bash scripts/stage1_pseudo_labels/visualize_pseudo_labels.sh \
#       --seq-ids sub10_clothesstand_000 sub11_largebox_001
set -euo pipefail

cd "$(dirname "$0")/../.."

DATA_DIR="/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano/omomo_correct_v2"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

python -m piano.inference.visualize_pseudo_labels \
    --data-dir "$DATA_DIR" \
    "${EXTRA_ARGS[@]}"
