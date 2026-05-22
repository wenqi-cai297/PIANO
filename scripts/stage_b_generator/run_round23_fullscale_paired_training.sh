#!/usr/bin/env bash
# Round-23 P1 full-scale Stage-2 paired training (with-plan vs no-plan).
#
# What this runs
# --------------
# Two configs in parallel (GPU0 + GPU1), same seed (42), same architecture,
# same data, same recipe — ONLY differences are
#   * plan_tokens_force_null: false (GPU0, with plan)
#   * plan_tokens_force_null: true  (GPU1, no plan; plan-aware losses zeroed)
#
# Decision rule (per analyses/2026-05-22_round23_phase1_alibi_attention_fix_report.md):
#   - with-plan ≫ no-plan on anchor_l2 / pos_full / plan_condition_diagnostics
#       → plan IS load-bearing; ALiBi + future routing fixes justified
#   - no-plan ≈ with-plan
#       → plan is redundant at this scale; simplify v26 by dropping plan branch
#
# Prereqs (server-side)
# ---------------------
#   * conda env `piano` activated (or override CONDA_ENV)
#   * git pulled to latest commit on the active branch
#   * cache/stage1_coarse_v1_full/normalization_train.json present
#   * Datasets at E:/Project/Datasets/InterAct/piano_official_process_4/{chairs,imhd,neuraldome,omomo_correct_v2}
#     (NOTE: server may have a different mount path — see configs' data.datasets section)
#
# Usage
# -----
#   bash scripts/stage_b_generator/run_round23_fullscale_paired_training.sh
#
# Output
# ------
# Per-config:
#   runs/training/stageB_anchordiff_v25_round23_{clean_alibi,noplan_clean_alibi}_FULL_DATA/
#     final.pt, best_val.pt, epoch_0020/0040/0060/0080.pt
#     loss_log.json, training_summary.json, metrics.jsonl, config.yaml
# Driver logs:
#   runs/training/round23_fullscale_launch_logs/{with_plan,no_plan}.log
# Sync tarball (created at the end):
#   runs/training/round23_fullscale_paired.tar.gz
#     — only logs + metrics + final.pt + best_val.pt (no intermediate epoch ckpts,
#        otherwise ~3GB; with final+best only it's ~2.5GB)

set -euo pipefail
cd "$(dirname "$0")/../.."

CONDA_ENV=${CONDA_ENV:-piano}
PY="conda run --no-capture-output -n ${CONDA_ENV} python"

LOG_DIR="runs/training/round23_fullscale_launch_logs"
mkdir -p "${LOG_DIR}"

# Auto-prefer the _local.yaml variant if present (server convention —
# Windows paths in the tracked config get replaced by Linux paths in
# the gitignored *_local.yaml on the server). See
# scripts/stage_b_generator/run_round23_make_local_configs.sh for one-line
# generation from the tracked variants.
CONFIG_WITH_PLAN_BASE="configs/training/anchordiff_v25_round23_clean_alibi_FULL_DATA.yaml"
CONFIG_WITH_PLAN_LOCAL="configs/training/anchordiff_v25_round23_clean_alibi_FULL_DATA_local.yaml"
CONFIG_NO_PLAN_BASE="configs/training/anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA.yaml"
CONFIG_NO_PLAN_LOCAL="configs/training/anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA_local.yaml"

if [[ -f "${CONFIG_WITH_PLAN_LOCAL}" ]]; then
    CONFIG_WITH_PLAN="${CONFIG_WITH_PLAN_LOCAL}"
    echo "[r23-fullscale] using local config: ${CONFIG_WITH_PLAN_LOCAL}"
else
    CONFIG_WITH_PLAN="${CONFIG_WITH_PLAN_BASE}"
    echo "[r23-fullscale] WARN: ${CONFIG_WITH_PLAN_LOCAL} not found — falling back to tracked base"
fi
if [[ -f "${CONFIG_NO_PLAN_LOCAL}" ]]; then
    CONFIG_NO_PLAN="${CONFIG_NO_PLAN_LOCAL}"
    echo "[r23-fullscale] using local config: ${CONFIG_NO_PLAN_LOCAL}"
else
    CONFIG_NO_PLAN="${CONFIG_NO_PLAN_BASE}"
    echo "[r23-fullscale] WARN: ${CONFIG_NO_PLAN_LOCAL} not found — falling back to tracked base"
fi

LOG_WITH_PLAN="${LOG_DIR}/with_plan.log"
LOG_NO_PLAN="${LOG_DIR}/no_plan.log"

OUT_WITH_PLAN="runs/training/stageB_anchordiff_v25_round23_clean_alibi_FULL_DATA"
OUT_NO_PLAN="runs/training/stageB_anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA"

echo "[r23-fullscale] repo root: $(pwd)"
echo "[r23-fullscale] log dir:   ${LOG_DIR}"
echo "[r23-fullscale] python:    $(which python)"
echo

# ─────────────────────────────────────────────────────────────────────
# Phase 0: preflight gate (run BEFORE either training starts)
# ─────────────────────────────────────────────────────────────────────

echo "===== Phase 0: preflight (unit tests + forward smoke on both configs) ====="
$PY -m pytest tests/test_stage2_stage1_coarse_condition.py -v \
    > "${LOG_DIR}/preflight_pytest.log" 2>&1
echo "  pytest:       see ${LOG_DIR}/preflight_pytest.log"

echo "  forward smoke with-plan..."
CUDA_VISIBLE_DEVICES=0 \
$PY -m piano.training.train_anchordiff \
    --config "${CONFIG_WITH_PLAN}" --smoke-test \
    > "${LOG_DIR}/preflight_smoke_with_plan.log" 2>&1
echo "    OK: ${LOG_DIR}/preflight_smoke_with_plan.log"

echo "  forward smoke no-plan..."
CUDA_VISIBLE_DEVICES=0 \
$PY -m piano.training.train_anchordiff \
    --config "${CONFIG_NO_PLAN}" --smoke-test \
    > "${LOG_DIR}/preflight_smoke_no_plan.log" 2>&1
echo "    OK: ${LOG_DIR}/preflight_smoke_no_plan.log"
echo

# ─────────────────────────────────────────────────────────────────────
# Phase 1: paired training (GPU0 with-plan, GPU1 no-plan, concurrent)
# ─────────────────────────────────────────────────────────────────────

echo "===== Phase 1: paired training launched concurrently on GPU0 + GPU1 ====="
echo "  with-plan   GPU0: ${OUT_WITH_PLAN}"
echo "  no-plan     GPU1: ${OUT_NO_PLAN}"

CUDA_VISIBLE_DEVICES=0 \
$PY -m piano.training.train_anchordiff \
    --config "${CONFIG_WITH_PLAN}" \
    > "${LOG_WITH_PLAN}" 2>&1 &
PID_WITH_PLAN=$!

CUDA_VISIBLE_DEVICES=1 \
$PY -m piano.training.train_anchordiff \
    --config "${CONFIG_NO_PLAN}" \
    > "${LOG_NO_PLAN}" 2>&1 &
PID_NO_PLAN=$!

echo "  PIDs: with-plan=${PID_WITH_PLAN}  no-plan=${PID_NO_PLAN}"
echo "  waiting for both to finish (~5-6h each at v18 scale)..."

set +e
wait "${PID_WITH_PLAN}"; RC_WITH=$?
wait "${PID_NO_PLAN}";   RC_NO=$?
set -e

echo "  exit codes: with-plan=${RC_WITH}  no-plan=${RC_NO}"
if [[ ${RC_WITH} -ne 0 || ${RC_NO} -ne 0 ]]; then
    echo "[r23-fullscale] FAIL — partial results in:"
    echo "  ${LOG_WITH_PLAN}"
    echo "  ${LOG_NO_PLAN}"
    # don't exit: still pack what we have so the user can inspect
fi
echo

# ─────────────────────────────────────────────────────────────────────
# Phase 2: tar.gz sync bundle
# ─────────────────────────────────────────────────────────────────────

echo "===== Phase 2: tarball ====="
TARBALL="runs/training/round23_fullscale_paired.tar.gz"
tar czf "${TARBALL}" \
    "${LOG_DIR}/" \
    "${OUT_WITH_PLAN}/final.pt" \
    "${OUT_WITH_PLAN}/best_val.pt" \
    "${OUT_WITH_PLAN}/metrics.jsonl" \
    "${OUT_WITH_PLAN}/loss_log.json" \
    "${OUT_WITH_PLAN}/training_summary.json" \
    "${OUT_WITH_PLAN}/config.yaml" \
    "${OUT_NO_PLAN}/final.pt" \
    "${OUT_NO_PLAN}/best_val.pt" \
    "${OUT_NO_PLAN}/metrics.jsonl" \
    "${OUT_NO_PLAN}/loss_log.json" \
    "${OUT_NO_PLAN}/training_summary.json" \
    "${OUT_NO_PLAN}/config.yaml" \
    2>/dev/null || true

echo "[r23-fullscale] tarball: ${TARBALL}"
ls -lh "${TARBALL}" 2>/dev/null | awk '{print "  size: "$5}'

echo
echo "[r23-fullscale] DONE"
echo
echo "To sync to local:"
echo "  scp <server>:$(pwd)/${TARBALL} runs/training/"
echo
echo "After download, run local diagnostics:"
echo "  bash scripts/stage_b_generator/run_round23_fullscale_local_diagnostics.sh"
