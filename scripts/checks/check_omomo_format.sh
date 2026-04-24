#!/usr/bin/env bash
# Inspect OMOMO (CHOIS processed_data) format after download.
# Verifies file structure, keys, shapes — catches format drift before training.
#
# Usage:
#   bash scripts/checks/check_omomo_format.sh
#   bash scripts/checks/check_omomo_format.sh --data-dir data/omomo/processed_data
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m piano.checks.omomo_format "$@"
