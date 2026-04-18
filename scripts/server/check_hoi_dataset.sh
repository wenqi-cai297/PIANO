#!/usr/bin/env bash
# Verify HOIDataset can load preprocessed OMOMO data.
#
# Usage:
#   bash scripts/server/check_hoi_dataset.sh
#   bash scripts/server/check_hoi_dataset.sh --data-dir /custom/path
set -euo pipefail

cd "$(dirname "$0")/../.."

DATA_DIR="/media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/piano"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

python -m piano.checks.hoi_dataset --data-dir "$DATA_DIR" "${EXTRA_ARGS[@]}"
