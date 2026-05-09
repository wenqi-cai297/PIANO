#!/usr/bin/env bash
# Extract v19 pseudo-labels (directional pelvis contact gate) for all 4
# InterAct subsets.
#
# v19 differences from v18:
#   --use-directional-pelvis-gate enables cylinder + upward-normal check
#     in process_sequence after mesh-distance and official-marker contact
#     signals are max-combined. Filters false-positive pelvis contacts
#     during dynamic motions (bat-swing, baseball-lift, etc.).
#   Output subdir: v19_h10_f05_pelvis20dir_official_semantic_marker
#
# Codex's caching infrastructure (mesh_cache, atlas_cache, seat-points cache,
# resume support for existing .npz) is fully reused — no new compute infra.
# Single-threaded but heavily-cached, ~30 min per subset on CPU.
#
# Local Windows defaults (override via env vars on other machines):
#   PIANO_ROOT=E:/Project/Datasets/InterAct/piano_official_process_4
#   INTERACT_ROOT=E:/Project/Datasets/InterAct/InterAct
#   OFFICIAL_ROOT=E:/Project/Datasets/InterAct/InterAct_official_process_4
#
# Usage:
#   bash scripts/stage1_pseudo_labels/extract_v19_pelvis_directional.sh
#   bash scripts/stage1_pseudo_labels/extract_v19_pelvis_directional.sh chairs
#   bash scripts/stage1_pseudo_labels/extract_v19_pelvis_directional.sh chairs imhd
set -euo pipefail

cd "$(dirname "$0")/../.."

PIANO_ROOT="${PIANO_ROOT:-E:/Project/Datasets/InterAct/piano_official_process_4}"
INTERACT_ROOT="${INTERACT_ROOT:-E:/Project/Datasets/InterAct/InterAct}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-E:/Project/Datasets/InterAct/InterAct_official_process_4}"
PYTHON="${PYTHON:-C:/Users/cwq29/miniforge3/envs/piano/python.exe}"
LABEL_SUBDIR="v19_h10_f05_pelvis20dir_official_semantic_marker"
SUBSETS=("chairs" "imhd" "neuraldome" "omomo_correct_v2")

if [[ $# -gt 0 ]]; then
    SUBSETS=("$@")
fi

for subset in "${SUBSETS[@]}"; do
    echo "=========================================================="
    echo "Extracting v19 pseudo-labels for InterAct subset: $subset"
    echo "=========================================================="
    data_dir="$PIANO_ROOT/$subset"
    mesh_dir="$INTERACT_ROOT/$subset/objects"
    output_dir="$data_dir/pseudo_labels/$LABEL_SUBDIR"

    if [[ ! -d "$data_dir" ]]; then
        echo "  [skip] $data_dir not found"
        continue
    fi
    if [[ ! -d "$mesh_dir" ]]; then
        echo "  [skip] $mesh_dir not found"
        continue
    fi
    mkdir -p "$output_dir"

    "$PYTHON" -m piano.data.pseudo_labels.run_all \
        --data-dir "$data_dir" \
        --mesh-dir "$mesh_dir" \
        --output-dir "$output_dir" \
        --mesh-suffixes "_face1000" "_simplified" "" \
        --official-interact-root "$OFFICIAL_ROOT" \
        --official-marker-hand-distance-m   0.10 \
        --official-marker-foot-distance-m   0.05 \
        --official-marker-pelvis-distance-m 0.20 \
        --use-directional-pelvis-gate
done

echo ""
echo "All v19 subsets done."
