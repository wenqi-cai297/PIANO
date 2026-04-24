#!/usr/bin/env bash
# Preprocess InterAct (4 subsets) → PIANO HumanML3D format.
# Each subset becomes its own PIANO data root under the output directory.
#
# Default paths (server):
#   InterAct: /media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/InterAct
#   SMPL-X:   <repo>/checkpoints/smpl_x_v1.1/models/smplx
#   Output:   /media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano
#
# Usage:
#   bash scripts/prep/preprocess_interact.sh                    # all 4 subsets
#   bash scripts/prep/preprocess_interact.sh --subset chairs    # one subset
#   bash scripts/prep/preprocess_interact.sh --num-samples-limit 10   # smoke test
set -euo pipefail

cd "$(dirname "$0")/../.."

INTERACT_DIR="/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/InterAct"
SMPLX_DIR="$(pwd)/checkpoints/smpl_x_v1.1/models/smplx"
OUTPUT_DIR="/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interact-dir) INTERACT_DIR="$2"; shift 2 ;;
        --smplx-dir) SMPLX_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

python -m piano.data.preprocess_interact \
    --interact-dir "$INTERACT_DIR" \
    --smplx-dir "$SMPLX_DIR" \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
