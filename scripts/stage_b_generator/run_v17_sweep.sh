#!/usr/bin/env bash
# v17 sweep: run v17-D (stacked) + v17-E.20 + v17-E.50 back-to-back on the
# same source ckpt. Each variant lands under its own EVAL_PREFIX so they
# can be diff'd cleanly afterward.
#
# Default source ckpt: v16 best_contact (matches the v17-C baseline that
# this sweep extends). Override SOURCE_RUN_DIR / SOURCE_CFG to swap.
#
# Skip individual variants by exporting SKIP_D=1 / SKIP_E20=1 / SKIP_E50=1.
#
# Per-variant cost on a single A6000 (matched 80-clip eval):
#   v17-D stacked  : ~70-90 min (per-step 10 inner × 10 outer + post-hoc 30 + GT roundtrip)
#   v17-E.20       : ~70-90 min (per-step 20 inner × 10 outer + GT roundtrip)
#   v17-E.50       : ~120-150 min (per-step 50 inner × 10 outer + GT roundtrip)
# Total ~4-6 hours.
#
# See analyses/2026-05-01_v17_per_step_result.md §"Next steps" for the
# decision rule that consumes these results, and
# analyses/2026-05-01_per_step_guidance_design.md §4 for the ablation matrix.

set -euo pipefail

export SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-runs/training/generator_v16_alignment_mirror}"
export SOURCE_CFG="${SOURCE_CFG:-configs/training/generator_v16_alignment_mirror.yaml}"
export CKPTS="${CKPTS:-best_contact}"
export NUM_CLIPS="${NUM_CLIPS:-80}"
export SEED="${SEED:-42}"
export SUMMARY_DETAIL="${SUMMARY_DETAIL:-compact}"
export DUMP_WANDB="${DUMP_WANDB:-0}"
export TRAIN="${TRAIN:-0}"
export GUIDANCE_RESIDUAL_SEED="${GUIDANCE_RESIDUAL_SEED:-42}"
export GUIDANCE_LAYERS="${GUIDANCE_LAYERS:-full_rvq}"
export GUIDANCE_LOSS="${GUIDANCE_LOSS:-target}"

run_variant() {
  local label="$1"
  local prefix="$2"
  local per_step="$3"
  local post_hoc="$4"

  echo
  echo "============================================================"
  echo "v17 sweep variant: ${label}"
  echo "  EVAL_PREFIX=${prefix}"
  echo "  PER_STEP_ITERS=${per_step}"
  echo "  GUIDANCE_STEPS=${post_hoc}"
  echo "  source ckpt: ${SOURCE_RUN_DIR}/${CKPTS}.pt"
  echo "============================================================"

  EVAL_PREFIX="${prefix}" \
  PER_STEP_ITERS="${per_step}" \
  GUIDANCE_STEPS="${post_hoc}" \
    bash scripts/stage_b_generator/run_v17_per_step_guidance.sh
}

if [[ "${SKIP_D:-0}" != "1" ]]; then
  run_variant "v17-D stacked (per_step=10, post_hoc=30; canonical MaskControl recipe)" \
    "stageB_v0_17_v16bc_stacked" 10 30
fi

if [[ "${SKIP_E20:-0}" != "1" ]]; then
  run_variant "v17-E.20 (per_step=20, no post-hoc)" \
    "stageB_v0_17_v16bc_per_step_iters20" 20 0
fi

if [[ "${SKIP_E50:-0}" != "1" ]]; then
  run_variant "v17-E.50 (per_step=50, no post-hoc)" \
    "stageB_v0_17_v16bc_per_step_iters50" 50 0
fi

echo
echo "============================================================"
echo "v17 sweep done. Sync these dirs back for analysis:"
echo "============================================================"
for prefix in stageB_v0_17_v16bc_stacked stageB_v0_17_v16bc_per_step_iters20 stageB_v0_17_v16bc_per_step_iters50; do
  echo "  runs/eval/${prefix}_bc_qual/{summary.json,full_guided/{summary.json,guidance_trace.json,generated.npz}}"
  echo "  runs/eval/${prefix}_bc_contact_dist/summary.json"
  echo "  runs/eval/${prefix}_bc_temporal_coupling/summary.json"
  echo "  runs/eval/${prefix}_bc_guided_temporal_coupling/summary.json"
  echo "  runs/eval/${prefix}_bc_alignment_to_gt_roundtrip/summary.json"
  echo "  runs/eval/${prefix}_bc_guided_alignment_to_gt_roundtrip/summary.json"
done
