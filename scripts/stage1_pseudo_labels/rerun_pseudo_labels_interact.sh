#!/usr/bin/env bash
# Clean and re-extract pseudo-labels for all InterAct subsets.
#
# What this does:
#   1. Kill any live tmux session matching $TMUX_SESSION (only if set).
#   2. Move the existing <subset>/pseudo_labels/ dirs to a timestamped
#      backup so nothing is destroyed.
#   3. Run pseudo_labels.run_all fresh for every subset.
#
# Usage (on server):
#   # All defaults (4 subsets, hardcoded server paths):
#   bash scripts/stage1_pseudo_labels/rerun_pseudo_labels_interact.sh
#
#   # Subset selection (positional after any flags):
#   bash scripts/stage1_pseudo_labels/rerun_pseudo_labels_interact.sh chairs
#
#   # Explicit dataset roots (CLI flags, preferred):
#   bash scripts/stage1_pseudo_labels/rerun_pseudo_labels_interact.sh \
#       --piano-root    /path/to/piano \
#       --interact-root /path/to/InterAct
#
#   # Combined flags + subsets:
#   bash scripts/stage1_pseudo_labels/rerun_pseudo_labels_interact.sh \
#       --piano-root /path/to/piano  chairs imhd
#
#   # Backup-suffix tag (default '<timestamp>_pre_phase3' for v5 / 3-class
#   # phase re-extraction; pass --backup-suffix to override):
#   bash scripts/stage1_pseudo_labels/rerun_pseudo_labels_interact.sh \
#       --backup-suffix 20260425_pre_v5_test
#
#   # Env-var fallback (still supported for back-compat):
#   PIANO_ROOT=/path/to/piano INTERACT_ROOT=/path/to/InterAct \
#       bash scripts/stage1_pseudo_labels/rerun_pseudo_labels_interact.sh
#
# Precedence: CLI flag > env var > hardcoded default.
set -euo pipefail

cd "$(dirname "$0")/../.."

# Hardcoded defaults (server). Used when neither CLI flag nor env var is set.
DEFAULT_PIANO_ROOT="/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano"
DEFAULT_INTERACT_ROOT="/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/InterAct"

PIANO_ROOT_CLI=""
INTERACT_ROOT_CLI=""
BACKUP_SUFFIX_CLI=""
SUBSETS=()

# --- Parse flags + subsets ---
# Flags consumed: --piano-root <path>, --interact-root <path>,
# --backup-suffix <tag>, -h/--help. Anything else is treated as a subset
# name (positional).
while [[ $# -gt 0 ]]; do
    case "$1" in
        --piano-root)
            PIANO_ROOT_CLI="$2"
            shift 2
            ;;
        --interact-root)
            INTERACT_ROOT_CLI="$2"
            shift 2
            ;;
        --backup-suffix)
            BACKUP_SUFFIX_CLI="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,33p' "$0"
            exit 0
            ;;
        --)
            shift
            SUBSETS+=("$@")
            break
            ;;
        --*)
            echo "Unknown flag: $1" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
        *)
            SUBSETS+=("$1")
            shift
            ;;
    esac
done

# Resolve roots: CLI flag > env var > hardcoded default.
PIANO_ROOT="${PIANO_ROOT_CLI:-${PIANO_ROOT:-$DEFAULT_PIANO_ROOT}}"
INTERACT_ROOT="${INTERACT_ROOT_CLI:-${INTERACT_ROOT:-$DEFAULT_INTERACT_ROOT}}"

if [[ ${#SUBSETS[@]} -eq 0 ]]; then
    SUBSETS=("chairs" "imhd" "neuraldome" "omomo_correct_v2")
fi

# TMUX_SESSION is intentionally unset by default. If the user is running
# this script from inside a tmux session, auto-killing "piano-labels"
# would frequently be the same session hosting the script. Only clean
# up a session when the caller explicitly names one to clean up, e.g.
#   TMUX_SESSION=old-piano-labels bash scripts/stage1_pseudo_labels/rerun_pseudo_labels_interact.sh
TMUX_SESSION="${TMUX_SESSION:-}"
# Backup suffix: CLI flag > timestamped default. The default tag
# '_pre_phase3' marks the v4→v5 re-extraction (5-class phase npz being
# replaced by 3-class). Pass --backup-suffix to override per-extraction.
BACKUP_SUFFIX="${BACKUP_SUFFIX_CLI:-$(date +%Y%m%d_%H%M%S)_pre_phase3}"

echo "Configuration:"
echo "  PIANO_ROOT     = $PIANO_ROOT"
echo "  INTERACT_ROOT  = $INTERACT_ROOT"
echo "  SUBSETS        = ${SUBSETS[*]}"
echo "  BACKUP_SUFFIX  = $BACKUP_SUFFIX"
echo ""

# --- Step 1: kill a named tmux session if the caller asked us to ---
if [[ -n "$TMUX_SESSION" ]] && command -v tmux >/dev/null 2>&1; then
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
