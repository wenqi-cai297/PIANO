#!/usr/bin/env bash
# v17-F Gumbel-noise ablation: A/B test the canonical-MaskControl
# Gumbel-Softmax / Concrete relaxation in the per-step inner loop
# vs PIANO's pre-v17-F pure-softmax expectation. Source-verified diff
# against `exitudio/ControlMM` documented in
# analyses/2026-05-01_per_step_guidance_design.md (route 1 / diff #2).
#
# Variants:
#   v17-F.10   per_step=10 Gumbel=ON   (matches MaskControl's `each_iter`)
#   v17-F.20   per_step=20 Gumbel=ON
#   v17-C-ng   per_step=10 Gumbel=OFF  (sanity: must reproduce v17-C 21.77 cm)
#   v17-E.20-ng per_step=20 Gumbel=OFF (sanity: must reproduce v17-E.20 18.62 cm)
#
# Skip individual variants by exporting SKIP_F10 / SKIP_F20 / SKIP_C_NG /
# SKIP_E20_NG = "1".
#
# Decision rule:
#   v17-F.10 ≥ v17-E.20 quality → Gumbel beats raw budget; ship Gumbel
#                                 + v17-F.10 (cheaper than v17-E.20).
#   v17-F.20 > v17-E.50 → Gumbel + budget compounds; consider v17-F.50.
#   v17-F.10 ≈ v17-C    → Gumbel is neutral on this generator; skip.
#
# Wallclock: ~1-2h per variant on a single A6000 (80-clip eval).

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
export GUIDANCE_STEPS="${GUIDANCE_STEPS:-0}"  # v17-F is per-step only

run_variant() {
  local label="$1"
  local prefix="$2"
  local per_step="$3"
  local gumbel="$4"

  echo
  echo "============================================================"
  echo "v17-F variant: ${label}"
  echo "  EVAL_PREFIX=${prefix}"
  echo "  PER_STEP_ITERS=${per_step}  PER_STEP_GUMBEL_SCALE=${gumbel}"
  echo "============================================================"

  EVAL_PREFIX="${prefix}" \
  PER_STEP_ITERS="${per_step}" \
  PER_STEP_GUMBEL_SCALE="${gumbel}" \
    bash scripts/stage_b_generator/run_v17_per_step_guidance.sh
}

# Gumbel-ON variants (v17-F).
if [[ "${SKIP_F10:-0}" != "1" ]]; then
  run_variant "v17-F.10 (per_step=10, Gumbel scale=1.0; canonical MaskControl)" \
    "stageB_v0_17_v16bc_f10_gumbel" 10 1.0
fi
if [[ "${SKIP_F20:-0}" != "1" ]]; then
  run_variant "v17-F.20 (per_step=20, Gumbel scale=1.0)" \
    "stageB_v0_17_v16bc_f20_gumbel" 20 1.0
fi

# Gumbel-OFF sanity checks (must reproduce existing v17-C/E.20 numbers).
if [[ "${SKIP_C_NG:-0}" != "1" ]]; then
  run_variant "v17-C-ng (per_step=10, Gumbel=OFF; sanity reproduces v17-C 21.77 cm)" \
    "stageB_v0_17_v16bc_c_no_gumbel" 10 0.0
fi
if [[ "${SKIP_E20_NG:-0}" != "1" ]]; then
  run_variant "v17-E.20-ng (per_step=20, Gumbel=OFF; sanity reproduces v17-E.20 18.62 cm)" \
    "stageB_v0_17_v16bc_e20_no_gumbel" 20 0.0
fi

echo
echo "============================================================"
echo "v17-F sweep done. Sync these dirs back for analysis:"
echo "============================================================"
for prefix in stageB_v0_17_v16bc_f10_gumbel stageB_v0_17_v16bc_f20_gumbel \
              stageB_v0_17_v16bc_c_no_gumbel stageB_v0_17_v16bc_e20_no_gumbel; do
  echo "  runs/eval/${prefix}_bc_qual/{summary.json,full_guided/{summary.json,guidance_trace.json,generated.npz}}"
  echo "  runs/eval/${prefix}_bc_contact_dist/summary.json"
  echo "  runs/eval/${prefix}_bc_temporal_coupling/summary.json"
  echo "  runs/eval/${prefix}_bc_guided_temporal_coupling/summary.json"
  echo "  runs/eval/${prefix}_bc_alignment_to_gt_roundtrip/summary.json"
  echo "  runs/eval/${prefix}_bc_guided_alignment_to_gt_roundtrip/summary.json"
done
