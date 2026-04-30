#!/usr/bin/env bash
# v17-G γ_int inference-boost ablation: scale every gamma_int parameter
# in-place by a constant immediately after model load, then run the
# v17-E.20 base config (per_step=20, Gumbel OFF, full_rvq post-hoc=0).
#
# Hypothesis (per analyses/2026-05-01_v17f_gumbel_result_and_p1_plan.md
# "P1"): D-A audit showed γ_int finished v14/v15/v16 training around
# 0.02 (~1/25 of typical ControlNet-style 0.5–1.0 → IntXAttn cross-
# attention is heavily underused). If γ_int is the contact-patch
# misalignment bottleneck, scaling it up at inference should improve
# alignment metrics monotonically until the IntXAttn output magnitude
# becomes OOD for the trained base path.
#
# Variants (default sweep):
#   v17-G.b1   boost=1.0   sanity reproducer of v17-E.20
#   v17-G.b2   boost=2.0   conservative (effective γ_int ≈ 0.04)
#   v17-G.b5   boost=5.0   moderate     (effective γ_int ≈ 0.10)
#   v17-G.b10  boost=10.0  aggressive   (effective γ_int ≈ 0.20)
#   v17-G.b20  boost=20.0  extreme      (effective γ_int ≈ 0.40, ≈ ControlNet typical)
#
# Skip individual variants by exporting SKIP_B1 / SKIP_B2 / SKIP_B5 /
# SKIP_B10 / SKIP_B20 = "1".
#
# Wallclock: 5 × ~80 min ≈ 7 h on a single A6000.
#
# Decision rule (analyses doc §"Decision rule for P1"):
#   monotone improvement → γ_int IS the bottleneck → P2 (re-init γ_int + finetune Stage B).
#   plateau at b5 / b10  → γ_int helps to a point. Ship best boost as additional config.
#   catastrophe at b20   → IntXAttn output is OOD for trained base path; do P3 architectural rework.
#   no monotonic trend   → γ_int is not the lever; pivot to OMOMO-style explicit contact target.

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
export GUIDANCE_STEPS="${GUIDANCE_STEPS:-0}"   # v17-G keeps per-step only
export PER_STEP_ITERS="${PER_STEP_ITERS:-20}"  # v17-E.20 base config
export PER_STEP_GUMBEL_SCALE="${PER_STEP_GUMBEL_SCALE:-0.0}"  # v17-F: Gumbel OFF on PIANO

run_variant() {
  local label="$1"
  local prefix="$2"
  local boost="$3"

  echo
  echo "============================================================"
  echo "v17-G variant: ${label}"
  echo "  EVAL_PREFIX=${prefix}"
  echo "  GAMMA_INT_BOOST=${boost}"
  echo "  PER_STEP_ITERS=${PER_STEP_ITERS}  PER_STEP_GUMBEL_SCALE=${PER_STEP_GUMBEL_SCALE}"
  echo "============================================================"

  EVAL_PREFIX="${prefix}" \
  GAMMA_INT_BOOST="${boost}" \
    bash scripts/stage_b_generator/run_v17_per_step_guidance.sh
}

if [[ "${SKIP_B1:-0}" != "1" ]]; then
  run_variant "v17-G.b1 (boost=1.0; sanity reproducer of v17-E.20)" \
    "stageB_v0_17_v16bc_g_b1" 1.0
fi
if [[ "${SKIP_B2:-0}" != "1" ]]; then
  run_variant "v17-G.b2 (boost=2.0; effective γ_int ≈ 0.04)" \
    "stageB_v0_17_v16bc_g_b2" 2.0
fi
if [[ "${SKIP_B5:-0}" != "1" ]]; then
  run_variant "v17-G.b5 (boost=5.0; effective γ_int ≈ 0.10)" \
    "stageB_v0_17_v16bc_g_b5" 5.0
fi
if [[ "${SKIP_B10:-0}" != "1" ]]; then
  run_variant "v17-G.b10 (boost=10.0; effective γ_int ≈ 0.20)" \
    "stageB_v0_17_v16bc_g_b10" 10.0
fi
if [[ "${SKIP_B20:-0}" != "1" ]]; then
  run_variant "v17-G.b20 (boost=20.0; effective γ_int ≈ 0.40 ≈ ControlNet typical)" \
    "stageB_v0_17_v16bc_g_b20" 20.0
fi

echo
echo "============================================================"
echo "v17-G sweep done. Sync these dirs back for analysis:"
echo "============================================================"
for prefix in stageB_v0_17_v16bc_g_b1 stageB_v0_17_v16bc_g_b2 stageB_v0_17_v16bc_g_b5 \
              stageB_v0_17_v16bc_g_b10 stageB_v0_17_v16bc_g_b20; do
  echo "  runs/eval/${prefix}_bc_qual/{summary.json,full_guided/{summary.json,guidance_trace.json,generated.npz}}"
  echo "  runs/eval/${prefix}_bc_contact_dist/summary.json"
  echo "  runs/eval/${prefix}_bc_temporal_coupling/summary.json"
  echo "  runs/eval/${prefix}_bc_guided_temporal_coupling/summary.json"
  echo "  runs/eval/${prefix}_bc_alignment_to_gt_roundtrip/summary.json"
  echo "  runs/eval/${prefix}_bc_guided_alignment_to_gt_roundtrip/summary.json"
done
