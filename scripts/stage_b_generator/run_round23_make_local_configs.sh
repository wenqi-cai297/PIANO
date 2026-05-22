#!/usr/bin/env bash
# Generate the gitignored _local.yaml variants of the Round-23 P1
# configs by rewriting the Windows dataset paths to the Linux server
# layout. The _local.yaml files are NOT tracked in git (per
# .gitignore rule `configs/training/*_local.yaml`).
#
# Usage (on the Linux server):
#   bash scripts/stage_b_generator/run_round23_make_local_configs.sh
#
# Override the data root via env:
#   DATASETS_ROOT=/path/to/InterAct/piano_official_process_4 \
#     bash scripts/stage_b_generator/run_round23_make_local_configs.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

DATASETS_ROOT=${DATASETS_ROOT:-/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano_official_process_4}
WIN_ROOT="E:/Project/Datasets/InterAct/piano_official_process_4"

if [[ ! -d "${DATASETS_ROOT}/chairs" ]]; then
    echo "ERROR: expected ${DATASETS_ROOT}/chairs to exist"
    echo "Set DATASETS_ROOT to the directory containing chairs/imhd/neuraldome/omomo_correct_v2"
    exit 1
fi

for VARIANT in clean_alibi noplan_clean_alibi; do
    SRC="configs/training/anchordiff_v25_round23_${VARIANT}_FULL_DATA.yaml"
    DST="configs/training/anchordiff_v25_round23_${VARIANT}_FULL_DATA_local.yaml"
    if [[ ! -f "${SRC}" ]]; then
        echo "ERROR: source config missing: ${SRC}"
        exit 1
    fi
    sed "s|${WIN_ROOT}|${DATASETS_ROOT}|g" "${SRC}" > "${DST}"
    echo "wrote ${DST}"
done

echo
echo "Done. Verify dataset paths:"
grep -E "root:" configs/training/anchordiff_v25_round23_clean_alibi_FULL_DATA_local.yaml
