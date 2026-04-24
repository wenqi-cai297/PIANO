#!/usr/bin/env bash
# Download OMOMO dataset (via the CHOIS processed_data release).
#
# Why CHOIS bundle: it contains the same raw sequences as OMOMO, but
# pre-processed into a cleaner format with text annotations, contact
# labels, and canonical object meshes already organized.
#
# Downloads:
#   1. processed_data.zip  — sequences + object meshes + text + contacts
#      (from https://github.com/lijiaman/chois_release)
#
# The user must separately obtain SMPL-X model files (not redistributable).
#
# Usage:
#   bash scripts/prep/download_omomo.sh [--output-dir data/omomo]
set -euo pipefail

OUTPUT_DIR="data/omomo"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")/../.."

echo "================================================================"
echo "Downloading OMOMO (via CHOIS processed_data) to: $OUTPUT_DIR"
echo "================================================================"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# CHOIS processed_data bundle: sequences + meshes + text + contacts
echo ""
echo "[1/1] Downloading CHOIS processed_data..."
gdown "https://drive.google.com/file/d/1ZG-9--RfUWj5oWYnvcONNuRuxaH_Zpw1/view?usp=sharing"

# The file may come down as either .tar.gz or .zip; handle both.
echo ""
echo "Extracting..."
if [ -f processed_data.tar.gz ]; then
    tar -xzf processed_data.tar.gz
    rm processed_data.tar.gz
elif [ -f processed_data.zip ]; then
    unzip -q processed_data.zip
    rm processed_data.zip
else
    echo "ERROR: expected processed_data.tar.gz or processed_data.zip not found" >&2
    ls -la
    exit 1
fi

echo ""
echo "================================================================"
echo "Download complete. Contents:"
echo "================================================================"
ls -la processed_data/ 2>/dev/null || ls -la .

echo ""
echo "NOTE: SMPL-X model files are required for FK (generating body meshes)."
echo "      Download SMPLX_MALE.npz / SMPLX_FEMALE.npz from https://smpl-x.is.tue.mpg.de"
echo "      and place them at: $OUTPUT_DIR/processed_data/smpl_all_models/smplx/"
echo ""
echo "Next step: run format inspection with:"
echo "  bash scripts/checks/check_omomo_format.sh --data-dir $OUTPUT_DIR/processed_data"
