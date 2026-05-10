#!/usr/bin/env bash
# PIANO-AnchorDiff v9 launcher (CondMDI inpainting + v-prediction).
# See analyses/ + CondMDI vault note for design context.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/training/anchordiff_v9_condmdi.yaml"

# Run inside the `piano` conda env so accelerate / piano are on PATH
# regardless of the parent shell's activation state.
RUN="${RUN:-conda run -n piano --no-capture-output}"

# PyTorch on Windows is built without libuv; the accelerate / torchrun
# rendezvous defaults to libuv and crashes otherwise. v8 + v9 both run
# single-GPU on this box, so we default to plain `python -m` and only
# go through `accelerate launch` when MULTI_GPU=1 (Linux only — the
# libuv workaround is fragile and N=10 subsample doesn't need 2× A6000).
export USE_LIBUV="${USE_LIBUV:-0}"

if [[ "${1:-}" == "--smoke-test" ]]; then
  echo "[anchordiff v9] smoke test mode: 1 batch through forward + backward"
  $RUN python -m piano.training.train_anchordiff --config "$CONFIG" --smoke-test
  exit 0
fi

if [[ "${MULTI_GPU:-0}" == "1" ]]; then
  ACCEL_CFG="${ACCEL_CFG:-configs/accelerate_config.yaml}"
  echo "[anchordiff v9] launching multi-GPU training (config: $ACCEL_CFG)"
  $RUN accelerate launch --config_file "$ACCEL_CFG" \
    -m piano.training.train_anchordiff --config "$CONFIG"
else
  echo "[anchordiff v9] launching single-GPU training (set MULTI_GPU=1 for accelerate)"
  $RUN python -m piano.training.train_anchordiff --config "$CONFIG"
fi
