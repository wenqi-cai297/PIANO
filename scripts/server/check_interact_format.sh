#!/usr/bin/env bash
# Inspect InterAct dataset format after unzip.
#
# Usage:
#   bash scripts/server/check_interact_format.sh
#   bash scripts/server/check_interact_format.sh --data-dir /custom/path
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m piano.checks.interact_format "$@"
