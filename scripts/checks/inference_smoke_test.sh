#!/usr/bin/env bash
# End-to-end inference smoke test with untrained PIANO + pretrained MoMask.
# Verifies the full generation pipeline runs cleanly before investing in training.
#
# Usage:
#   bash scripts/checks/inference_smoke_test.sh
#   bash scripts/checks/inference_smoke_test.sh --num-samples 8 --device cuda
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m piano.checks.inference_smoke_test "$@"
