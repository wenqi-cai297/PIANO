#!/usr/bin/env bash
# Extract pseudo interaction labels for preprocessed OMOMO data.
# CPU-only geometric computation — takes ~a few hours for 4919 sequences.
#
# Default paths (server):
#   data-dir:    /media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/piano
#   mesh-dir:    /media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/processed_data/captured_objects
#   output-dir:  /media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/piano/pseudo_labels
#
# Usage:
#   bash scripts/data/extract_pseudo_labels_omomo.sh
#   bash scripts/data/extract_pseudo_labels_omomo.sh --data-dir X --mesh-dir Y --output-dir Z
set -euo pipefail

cd "$(dirname "$0")/../.."

DATA_DIR="/media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/piano"
MESH_DIR="/media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/processed_data/captured_objects"
OUTPUT_DIR="/media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/piano/pseudo_labels"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        --mesh-dir)   MESH_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

python -m piano.data.pseudo_labels.run_all \
    --data-dir "$DATA_DIR" \
    --mesh-dir "$MESH_DIR" \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
