#!/usr/bin/env bash
# PIANO-AnchorDiff v1 launcher.
# See analyses/2026-05-08_piano_anchordiff_design.md for design context.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/training/anchordiff_v1.yaml"

if [[ "${1:-}" == "--smoke-test" ]]; then
  echo "[anchordiff] smoke test mode: 1 batch through forward + backward"
  python -m piano.training.train_anchordiff --config "$CONFIG" --smoke-test
  exit 0
fi

# Real training run via Accelerate.
ACCEL_CFG="${ACCEL_CFG:-configs/accelerate_config.yaml}"
echo "[anchordiff] launching multi-GPU training (config: $ACCEL_CFG)"
accelerate launch --config_file "$ACCEL_CFG" \
  -m piano.training.train_anchordiff --config "$CONFIG"
