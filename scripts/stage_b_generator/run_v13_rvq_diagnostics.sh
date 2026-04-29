#!/usr/bin/env bash
set -euo pipefail

# Stage B v13 path diagnostics:
#   1) generate soft/hard and mixed-RVQ condition dirs
#   2) score contact distance
#   3) score temporal coupling
#
# Override examples:
#   CKPT=runs/training/generator_v13_target_trajectory_contact/final.pt \
#     bash scripts/stage_b_generator/run_v13_rvq_diagnostics.sh
#   NUM_CLIPS=20 bash scripts/stage_b_generator/run_v13_rvq_diagnostics.sh

CFG="${CFG:-configs/training/generator_v13_target_trajectory_contact.yaml}"
CKPT="${CKPT:-runs/training/generator_v13_target_trajectory_contact/best_val.pt}"
OUT_DIR="${OUT_DIR:-runs/eval/stageB_v0_13_rvq_path_diagnostics_bv}"
NUM_CLIPS="${NUM_CLIPS:-80}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
W_TEXT="${W_TEXT:-4.0}"
W_INT="${W_INT:-2.0}"
RESIDUAL_SEED="${RESIDUAL_SEED:-1234}"
SUMMARY_DETAIL="${SUMMARY_DETAIL:-compact}"

conditions=(
  soft_train_full
  hard_train_argmax_full
  hard_train_argmax_gt_residual
  mixed_gt_all
  mixed_pred_all
  mixed_gt_base_pred_residual
  mixed_pred_base_gt_residual
)

echo "============================================================"
echo "[diagnose] ${CKPT}"
echo "============================================================"
python scripts/stage_b_generator/diagnose_rvq_paths.py \
  --config "$CFG" \
  --ckpt "$CKPT" \
  --output-dir "$OUT_DIR" \
  --num-clips "$NUM_CLIPS" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --w-text "$W_TEXT" \
  --w-int "$W_INT" \
  --residual-seed "$RESIDUAL_SEED" \
  --detail "$SUMMARY_DETAIL"

contact_args=()
temporal_args=()
for cond in "${conditions[@]}"; do
  contact_args+=(--input-dir "${OUT_DIR}/${cond}")
  temporal_args+=(--input-dir "${OUT_DIR}/${cond}")
done

echo
echo "============================================================"
echo "[measure] contact distance"
echo "============================================================"
python scripts/stage_b_generator/measure_contact_distance.py \
  "${contact_args[@]}" \
  --output-dir "${OUT_DIR}/contact_dist" \
  --detail "$SUMMARY_DETAIL"

echo
echo "============================================================"
echo "[measure] temporal coupling"
echo "============================================================"
python scripts/stage_b_generator/measure_temporal_coupling.py \
  "${temporal_args[@]}" \
  --output-dir "${OUT_DIR}/temporal_coupling" \
  --fps 20 \
  --coupling-threshold 0.5 \
  --moving-speed-threshold 0.15 \
  --detail "$SUMMARY_DETAIL"

echo
echo "Done:"
echo "  ${OUT_DIR}/diagnostic_summary.json"
echo "  ${OUT_DIR}/contact_dist/summary.json"
echo "  ${OUT_DIR}/temporal_coupling/summary.json"
