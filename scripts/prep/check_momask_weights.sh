#!/usr/bin/env bash
# Sanity check: verify MoMask pretrained weights load via our adapter.
# Single most important pre-training check.
#
# Usage:
#   bash scripts/prep/check_momask_weights.sh
#   bash scripts/prep/check_momask_weights.sh --ckpt-root path/to/t2m --device cuda
set -euo pipefail

cd "$(dirname "$0")/../.."  # project root

python -m piano.checks.momask_weights "$@"
