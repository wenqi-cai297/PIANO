#!/usr/bin/env bash
# Round-28 server-path config translator. Rewrites Windows dataset
# paths to the Linux server layout, producing gitignored _local.yaml
# variants for every R28 config.
#
# Usage (on the Linux server):
#   bash scripts/stage_b_generator/run_round28_make_local_configs.sh
#
# Override the data root via env:
#   DATASETS_ROOT=/path/to/InterAct/piano_official_process_4 \
#     bash scripts/stage_b_generator/run_round28_make_local_configs.sh

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
    "anchordiff_r28_a0_input_add_48clip"
    "anchordiff_r28_a1_gated_input_48clip"
    "anchordiff_r28_a2_per_layer_adapter_48clip"
    "anchordiff_r28_a3_best_long_48clip"
    "anchordiff_r28_b0_baseline_48clip"
    "anchordiff_r28_b1_interaction_only_48clip"
    "anchordiff_r28_b2_body_only_all_on_48clip"
    "anchordiff_r28_b3_body_only_energy_48clip"
    "anchordiff_r28_b4_interaction_plus_body_48clip"
    "anchordiff_r28_c1_hints_plus_gait_48clip"
    "anchordiff_r28_c2_hints_plus_hint_consistency_48clip"
    "anchordiff_r28_c3_hints_gait_consistency_48clip"
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
grep -E "root:" configs/training/anchordiff_r28_a0_input_add_48clip_local.yaml
