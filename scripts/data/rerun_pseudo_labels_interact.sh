#!/usr/bin/env bash
# Clean and re-extract pseudo-labels for all InterAct subsets, after the P0
# fixes (deterministic per-object patch atlas + correct fps propagation).
#
# What this does:
#   1. Kill any live tmux session matching $TMUX_SESSION (default: piano-labels).
#   2. Move the existing <subset>/pseudo_labels/ dirs to a timestamped
#      backup so nothing is destroyed. (Old .npz files were extracted with
#      fps=30 default + random patch ids — not trustworthy.)
#   3. Run pseudo_labels.run_all fresh for every subset.
#
# Usage (on server):
#   bash scripts/data/rerun_pseudo_labels_interact.sh
#   bash scripts/data/rerun_pseudo_labels_interact.sh chairs       # one subset
#   TMUX_SESSION=my-run bash scripts/data/rerun_pseudo_labels_interact.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

PIANO_ROOT="/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano"
INTERACT_ROOT="/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/InterAct"
SUBSETS=("chairs" "imhd" "neuraldome" "omomo_correct_v2")
TMUX_SESSION="${TMUX_SESSION:-piano-labels}"
BACKUP_SUFFIX="$(date +%Y%m%d_%H%M%S)_pre_fps_fix"

if [[ $# -gt 0 ]]; then
    SUBSETS=("$@")
fi

# --- Step 1: kill live tmux session if present ---
if command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "Killing tmux session: $TMUX_SESSION"
        tmux kill-session -t "$TMUX_SESSION"
    else
        echo "No tmux session named '$TMUX_SESSION' to kill."
    fi
fi

# --- Step 2: move existing pseudo_labels dirs to backup ---
for subset in "${SUBSETS[@]}"; do
    pl_dir="$PIANO_ROOT/$subset/pseudo_labels"
    if [[ -d "$pl_dir" ]]; then
        backup="${pl_dir}.${BACKUP_SUFFIX}"
        echo "Backing up $pl_dir -> $backup"
        mv "$pl_dir" "$backup"
    fi
done

# --- Step 3: re-run extraction ---
for subset in "${SUBSETS[@]}"; do
    echo "=========================================================="
    echo "Re-extracting pseudo-labels for subset: $subset"
    echo "=========================================================="
    data_dir="$PIANO_ROOT/$subset"
    mesh_dir="$INTERACT_ROOT/$subset/objects"
    output_dir="$data_dir/pseudo_labels"

    if [[ ! -d "$data_dir" ]]; then
        echo "  [skip] $data_dir not found (run preprocess_interact first)"
        continue
    fi

    # fps is auto-resolved from <data_dir>/summary.json or <PIANO_ROOT>/summary.json.
    # Patch atlas is now deterministic per object_id and cached under
    # <output_dir>/patch_atlas/.
    python -m piano.data.pseudo_labels.run_all \
        --data-dir "$data_dir" \
        --mesh-dir "$mesh_dir" \
        --output-dir "$output_dir" \
        --mesh-suffixes "_face1000" "_simplified" ""
done

echo ""
echo "All subsets done. Backups kept under *.${BACKUP_SUFFIX}."
