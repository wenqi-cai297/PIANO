#!/usr/bin/env bash
# Round-27 Tier-0A server-path config translator. Rewrites Windows
# dataset paths to the Linux server layout, producing gitignored
# _local.yaml variants for the six Tier-0 configs.
#
# Usage (on the Linux server):
#   bash scripts/stage_b_generator/run_round27_make_local_configs.sh
#
# Override the data root via env:
#   DATASETS_ROOT=/path/to/InterAct/piano_official_process_4 \
#     bash scripts/stage_b_generator/run_round27_make_local_configs.sh

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
    "anchordiff_t0a1_hand_oracle_hint_48clip"
    "anchordiff_t0a2_foot_oracle_hint_48clip"
    "anchordiff_t0a3_full_oracle_hint_48clip"
    "anchordiff_t0b1_temporal_losses_48clip_from_v27"
    "anchordiff_t0b2_temporal_losses_48clip_from_r23"
    "anchordiff_t0ab_full_oracle_hint_temporal_losses_48clip"
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
grep -E "root:" configs/training/anchordiff_t0a1_hand_oracle_hint_48clip_local.yaml
