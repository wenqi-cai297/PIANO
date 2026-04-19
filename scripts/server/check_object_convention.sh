#!/usr/bin/env bash
# Verify object pose convention + inverse-transform correctness.
#
# Pre-requirement: run preprocess_interact AFTER the object_rotations save
# was added to preprocess_interact.py. Old preprocessed data has no
# object_rotations and this check will refuse to run.
#
# Usage:
#   bash scripts/server/check_object_convention.sh              # defaults to chairs
#   bash scripts/server/check_object_convention.sh \
#        --data-dir /media/.../InterAct/piano/imhd \
#        --mesh-dir /media/.../InterAct/InterAct/imhd/objects
set -euo pipefail

cd "$(dirname "$0")/../.."

DATA_DIR="/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano/chairs"
MESH_DIR="/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/InterAct/chairs/objects"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --mesh-dir) MESH_DIR="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

python -m piano.checks.object_convention \
    --data-dir "$DATA_DIR" \
    --mesh-dir "$MESH_DIR" \
    "${EXTRA_ARGS[@]}"
