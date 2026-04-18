#!/usr/bin/env bash
# Preprocess OMOMO (CHOIS format) into PIANO-ready HumanML3D 263-dim data.
#
# Default paths assume the standard server layout:
#   Datasets:  /media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/processed_data
#   SMPL-X:    <repo>/checkpoints/smpl_x_v1.1/models/smplx
#   Output:    <repo>/data/omomo
#
# Override any of them on the command line, e.g.:
#   bash scripts/data/preprocess_omomo.sh \
#       --omomo-dir /custom/path/processed_data \
#       --smplx-dir /custom/path/smplx \
#       --output-dir /custom/path/out \
#       --device cuda
set -euo pipefail

cd "$(dirname "$0")/../.."

OMOMO_DIR="/media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/processed_data"
SMPLX_DIR="$(pwd)/checkpoints/smpl_x_v1.1/models/smplx"
OUTPUT_DIR="/media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/piano"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --omomo-dir) OMOMO_DIR="$2"; shift 2 ;;
        --smplx-dir) SMPLX_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

python -m piano.data.preprocess_omomo \
    --omomo-dir "$OMOMO_DIR" \
    --smplx-dir "$SMPLX_DIR" \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
