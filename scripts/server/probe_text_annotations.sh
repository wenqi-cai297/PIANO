#!/usr/bin/env bash
# One-shot probe: dump InterAct text.txt samples + annotation CSV structure
# so we can pick the right parser for action_segment_sweep.
#
# Output: runs/checks/text_annotations/<ts>/{summary.json,preview.md}
#
# Usage:
#   bash scripts/server/probe_text_annotations.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

INTERACT_ROOT="${INTERACT_ROOT:-/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/InterAct}"

python -m piano.checks.probe_text_annotations \
    --interact-dir "$INTERACT_ROOT"
