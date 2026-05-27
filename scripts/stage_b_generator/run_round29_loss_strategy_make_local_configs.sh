#!/usr/bin/env bash
# Round-29 loss-strategy server-path config translator. Rewrites Windows
# dataset paths to the Linux server layout, producing gitignored
# _local.yaml variants for every R29 loss-strategy full-data config.
#
# Usage (on the Linux server):
#   bash scripts/stage_b_generator/run_round29_loss_strategy_make_local_configs.sh
#
# Override the data root via env:
#   DATASETS_ROOT=/path/to/InterAct/piano_official_process_4 \
#     bash scripts/stage_b_generator/run_round29_loss_strategy_make_local_configs.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

DATASETS_ROOT=${DATASETS_ROOT:-/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano_official_process_4}
WIN_ROOT="E:/Project/Datasets/InterAct/piano_official_process_4"

if [[ ! -d "${DATASETS_ROOT}/chairs" ]]; then
    echo "ERROR: expected ${DATASETS_ROOT}/chairs to exist"
    echo "Set DATASETS_ROOT to the directory containing chairs/imhd/neuraldome/omomo_correct_v2"
    exit 1
fi

CONFIGS=(
    "anchordiff_r29_lsf_a2_baseline_from_scratch"
    "anchordiff_r29_lsf_a3_baseline_from_scratch"
    "anchordiff_r29_lsf_a2_anchor2_mixed"
    "anchordiff_r29_lsf_a3_anchor2_mixed"
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
echo "Done. Verify dataset paths in one of the _local.yaml variants:"
grep -E "root:" configs/training/anchordiff_r29_lsf_a3_baseline_from_scratch_local.yaml
