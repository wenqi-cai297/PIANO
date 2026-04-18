#!/usr/bin/env bash
# Render skeleton motion videos (mp4) from real or generated motion sources.
#
# Usage:
#   # Real OMOMO samples (first 4)
#   bash scripts/server/visualize_motion.sh real \
#       --data-dir /media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/piano
#
#   # Generated motion from a smoke-test run
#   bash scripts/server/visualize_motion.sh generated \
#       --run-dir runs/checks/inference_smoke_test/2026-04-19_063940
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m piano.inference.visualize_motion "$@"
