#!/usr/bin/env bash
# Round-23 P1 local diagnostics — run after extracting the server tarball.
#
# Prerequisites (local-side):
#   1. tar -xzf runs/training/round23_fullscale_paired.tar.gz -C ./
#      (extracts: runs/training/stageB_anchordiff_v25_round23_*_FULL_DATA/
#                 runs/training/round23_fullscale_launch_logs/)
#   2. conda env `piano` available
#   3. local Stage-1 cache present at cache/stage1_coarse_v1_full/
#      (the diagnostics need normalization_train.json for stage1_coarse)
#
# What this runs (per ckpt):
#   1. plan_condition_diagnostics.py  — §7.4 plan sensitivity + route sensitivity + conflict cases
#   2. plan_cross_attention_inspector.py — attention heatmaps + entropy + top1=nearest metrics
#
# Then writes a side-by-side comparison report:
#   analyses/2026-05-XX_round23_fullscale_results.md
#
# Wallclock: ~30 min on cuda:0 (full-DDPM sampling × ~14 plan + 4 route + 3 conflict
#                               variants × 2 ckpts).

set -euo pipefail
cd "$(dirname "$0")/../.."

CONDA_ENV=${CONDA_ENV:-piano}
PY="conda run --no-capture-output -n ${CONDA_ENV} python"

WITH_PLAN_OUT="runs/training/stageB_anchordiff_v25_round23_clean_alibi_FULL_DATA"
NO_PLAN_OUT="runs/training/stageB_anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA"

CONFIG_WITH_PLAN="configs/training/anchordiff_v25_round23_clean_alibi_FULL_DATA.yaml"
CONFIG_NO_PLAN="configs/training/anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA.yaml"

WITH_PLAN_CKPT="${WITH_PLAN_OUT}/final.pt"
NO_PLAN_CKPT="${NO_PLAN_OUT}/final.pt"

for f in "${WITH_PLAN_CKPT}" "${NO_PLAN_CKPT}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: missing ckpt at ${f}"
        echo "  Did you extract the server tarball into ./ ?"
        exit 1
    fi
done

OUT_DIR="analyses/round23_fullscale_diagnostics"
mkdir -p "${OUT_DIR}"

echo "===== Round-23 fullscale local diagnostics ====="
echo "  with-plan ckpt: ${WITH_PLAN_CKPT}"
echo "  no-plan ckpt:   ${NO_PLAN_CKPT}"
echo "  output dir:     ${OUT_DIR}"
echo

# ─────────────────────────────────────────────────────────────────────
# Phase 1: plan_condition_diagnostics on both ckpts
# ─────────────────────────────────────────────────────────────────────

echo "===== Phase 1: plan_condition_diagnostics ====="

echo "  with-plan..."
$PY scripts/stage_b_generator/plan_condition_diagnostics.py \
    --config "${CONFIG_WITH_PLAN}" \
    --ckpt   "${WITH_PLAN_CKPT}" \
    --output "${OUT_DIR}/with_plan_diagnostic.json" \
    --md     "${OUT_DIR}/with_plan_diagnostic.md" \
    --bucket train --clip-idx 0 --cfg-scale 1.0 --seed 42 \
    > "${OUT_DIR}/with_plan_diagnostic.run.log" 2>&1
echo "    → ${OUT_DIR}/with_plan_diagnostic.{json,md}"

echo "  no-plan..."
$PY scripts/stage_b_generator/plan_condition_diagnostics.py \
    --config "${CONFIG_NO_PLAN}" \
    --ckpt   "${NO_PLAN_CKPT}" \
    --output "${OUT_DIR}/no_plan_diagnostic.json" \
    --md     "${OUT_DIR}/no_plan_diagnostic.md" \
    --bucket train --clip-idx 0 --cfg-scale 1.0 --seed 42 \
    > "${OUT_DIR}/no_plan_diagnostic.run.log" 2>&1
echo "    → ${OUT_DIR}/no_plan_diagnostic.{json,md}"
echo

# ─────────────────────────────────────────────────────────────────────
# Phase 2: plan_cross_attention_inspector on both ckpts
# ─────────────────────────────────────────────────────────────────────

echo "===== Phase 2: plan_cross_attention_inspector ====="

echo "  with-plan attention..."
$PY scripts/stage_b_generator/plan_cross_attention_inspector.py \
    --config "${CONFIG_WITH_PLAN}" \
    --ckpt   "${WITH_PLAN_CKPT}" \
    --output "${OUT_DIR}/with_plan_attention" \
    --bucket train --clip-idx 0 --t-step 200 --seed 42 \
    > "${OUT_DIR}/with_plan_attention.run.log" 2>&1
echo "    → ${OUT_DIR}/with_plan_attention{.json, __all_layers.png}"

echo "  no-plan attention (sanity check — slopes should be untrained baseline since plan was masked)..."
$PY scripts/stage_b_generator/plan_cross_attention_inspector.py \
    --config "${CONFIG_NO_PLAN}" \
    --ckpt   "${NO_PLAN_CKPT}" \
    --output "${OUT_DIR}/no_plan_attention" \
    --bucket train --clip-idx 0 --t-step 200 --seed 42 \
    > "${OUT_DIR}/no_plan_attention.run.log" 2>&1
echo "    → ${OUT_DIR}/no_plan_attention{.json, __all_layers.png}"
echo

# ─────────────────────────────────────────────────────────────────────
# Phase 3: brief stdout comparison summary
# ─────────────────────────────────────────────────────────────────────

echo "===== Phase 3: comparison summary ====="
$PY -c "
import json
from pathlib import Path

for name, path in [
    ('with-plan', '${OUT_DIR}/with_plan_diagnostic.json'),
    ('no-plan',   '${OUT_DIR}/no_plan_diagnostic.json'),
]:
    d = json.loads(Path(path).read_text('utf-8'))
    m = d.get('metrics', {})
    gt = m.get('gt', {}).get('far_unobserved_error_cm', float('nan'))
    zr = m.get('zero', {}).get('far_unobserved_error_cm', float('nan'))
    wr = m.get('wrong_clip', {}).get('far_unobserved_error_cm', float('nan'))
    ac = m.get('gt', {}).get('plan_anchor_contact_realization_cm', float('nan'))
    print(f'{name:>10}  gt={gt:6.2f}  zero={zr:6.2f}  wrong={wr:6.2f}  '
          f'gt-zero gap={zr-gt:+6.2f} cm  anchor_realization_gt={ac:6.2f} cm')

print()
print('Pass criteria (per piano_interaction_plan_pipeline_reframe.md §7.4 + §9.2):')
print('  §7.4 plan sensitivity: gt-zero gap >= 5 cm')
print('  anchor realization:    < 20 cm at GT plan')
print('  transition vel jump:   < 3 cm/frame (in JSON pass_gates)')
"
echo
echo "[r23-local-diag] DONE — see ${OUT_DIR}/*.md for full per-clip tables."
