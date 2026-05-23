#!/usr/bin/env bash
# Round-26 v27 server-path config translator. Rewrites Windows dataset
# paths in the v27 config to the Linux server layout, producing the
# gitignored _local.yaml variant.
#
# Usage (on the Linux server):
#   bash scripts/stage_b_generator/run_round26_make_local_configs.sh
#
# Override the data root via env:
#   DATASETS_ROOT=/path/to/InterAct/piano_official_process_4 \
#     bash scripts/stage_b_generator/run_round26_make_local_configs.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

DATASETS_ROOT=${DATASETS_ROOT:-/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano_official_process_4}
WIN_ROOT="E:/Project/Datasets/InterAct/piano_official_process_4"

if [[ ! -d "${DATASETS_ROOT}/chairs" ]]; then
    echo "ERROR: expected ${DATASETS_ROOT}/chairs to exist"
    echo "Set DATASETS_ROOT to the directory containing chairs/imhd/neuraldome/omomo_correct_v2"
    exit 1
fi

# v27 Round-26 motion-faithful fine-tune.
CONFIGS=(
    "anchordiff_v27_stage2_anchoraware_FULL_DATA"
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
echo "Done. Verify dataset paths in v27 _local.yaml:"
grep -E "root:" configs/training/anchordiff_v27_stage2_anchoraware_FULL_DATA_local.yaml
