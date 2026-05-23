#!/usr/bin/env bash
# Round-25 P0 diagnostic-bundle config translator for the Linux server.
# Rewrites Windows dataset paths in Round-25 configs (v26 + D4 + D5
# variants) to the server layout, producing the gitignored
# _local.yaml variants.
#
# Usage (on the Linux server):
#   bash scripts/stage_b_generator/run_round25_make_local_configs.sh
#
# Override the data root via env:
#   DATASETS_ROOT=/path/to/InterAct/piano_official_process_4 \
#     bash scripts/stage_b_generator/run_round25_make_local_configs.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

DATASETS_ROOT=${DATASETS_ROOT:-/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano_official_process_4}
WIN_ROOT="E:/Project/Datasets/InterAct/piano_official_process_4"

if [[ ! -d "${DATASETS_ROOT}/chairs" ]]; then
    echo "ERROR: expected ${DATASETS_ROOT}/chairs to exist"
    echo "Set DATASETS_ROOT to the directory containing chairs/imhd/neuraldome/omomo_correct_v2"
    exit 1
fi

# v26 mainline + D4 overfit configs + D5 loss-weight variants.
CONFIGS=(
    "anchordiff_v26_FULL_DATA"
    "anchordiff_v26_d4_overfit8"
    "anchordiff_v26_d4_overfit16"
    "anchordiff_v26_d5_v0_baseline"
    "anchordiff_v26_d5_v1_hand2x_foot2x"
    "anchordiff_v26_d5_v2_hand5x_foot5x"
    # Round-25 P1.2 — D5 30-epoch redo variants (5-ep originals had
    # only 40 grad steps, too short to test H2).
    "anchordiff_v26_d5_v0_baseline_30ep"
    "anchordiff_v26_d5_v1_hand2x_foot2x_30ep"
    "anchordiff_v26_d5_v2_hand5x_foot5x_30ep"
)

for NAME in "${CONFIGS[@]}"; do
    SRC="configs/training/${NAME}.yaml"
    DST="configs/training/${NAME}_local.yaml"
    if [[ ! -f "${SRC}" ]]; then
        echo "ERROR: source config missing: ${SRC}"
        exit 1
    fi
    sed "s|${WIN_ROOT}|${DATASETS_ROOT}|g" "${SRC}" > "${DST}"
    echo "wrote ${DST}"
done

echo
echo "Done. Verify dataset paths in v26 _local.yaml:"
grep -E "root:" configs/training/anchordiff_v26_FULL_DATA_local.yaml
