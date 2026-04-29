#!/usr/bin/env bash
# Train and evaluate Stage B v0.13 target-trajectory contact loss.
#
# Useful overrides:
#   TRAIN=0 bash scripts/stage_b_generator/run_v13_target_trajectory.sh
#   EVAL=0 bash scripts/stage_b_generator/run_v13_target_trajectory.sh
#   CKPTS="best_contact final" bash scripts/stage_b_generator/run_v13_target_trajectory.sh
#   NUM_CLIPS=80 SEED=42 bash scripts/stage_b_generator/run_v13_target_trajectory.sh

set -euo pipefail

CFG="${CFG:-configs/training/generator_v13_target_trajectory_contact.yaml}"
RUN_DIR="${RUN_DIR:-runs/training/generator_v13_target_trajectory_contact}"
RUN_NAME="${RUN_NAME:-predictor_stageB_v13_target_trajectory_contact}"
EVAL_PREFIX="${EVAL_PREFIX:-stageB_v0_13_target_trajectory}"
NUM_CLIPS="${NUM_CLIPS:-80}"
SEED="${SEED:-42}"
CKPTS="${CKPTS:-best_contact best_val final}"
TRAIN="${TRAIN:-1}"
EVAL="${EVAL:-1}"
DUMP_WANDB="${DUMP_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-piano}"

if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG" >&2
  exit 1
fi

mkdir -p "$RUN_DIR" runs/wandb_logs

if [[ "$TRAIN" == "1" ]]; then
  echo
  echo "============================================================"
  echo "[train] $CFG"
  echo "============================================================"
  accelerate launch --config_file configs/accelerate_config.yaml \
    -m piano.training.train_generator \
    --config "$CFG" \
    2>&1 | tee "${RUN_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
else
  echo "TRAIN=0, skipping training."
fi

if [[ "$EVAL" == "1" ]]; then
  gt_dir="runs/eval/${EVAL_PREFIX}_gt_roundtrip_${NUM_CLIPS}"
  if [[ ! -d "${gt_dir}/gt_original" || ! -d "${gt_dir}/gt_roundtrip" ]]; then
    echo
    echo "============================================================"
    echo "[eval] GT VQ roundtrip refs -> ${gt_dir}"
    echo "============================================================"
    python scripts/stage_b_generator/gt_vq_roundtrip.py \
      --config "$CFG" \
      --num-clips "$NUM_CLIPS" \
      --seed "$SEED" \
      --output-dir "$gt_dir"
  else
    echo "[eval] Reusing GT VQ roundtrip refs: ${gt_dir}"
  fi

  read -r -a CKPT_ARRAY <<< "$CKPTS"
  for ckpt_name in "${CKPT_ARRAY[@]}"; do
    ckpt_path="${RUN_DIR}/${ckpt_name}.pt"
    if [[ ! -f "$ckpt_path" ]]; then
      echo "WARN: missing checkpoint, skipping eval: ${ckpt_path}" >&2
      continue
    fi

    case "$ckpt_name" in
      best_contact) ckpt_tag="bc" ;;
      best_val) ckpt_tag="bv" ;;
      final) ckpt_tag="final" ;;
      *) ckpt_tag="$ckpt_name" ;;
    esac

    qual_dir="runs/eval/${EVAL_PREFIX}_${ckpt_tag}_qual"
    dist_dir="runs/eval/${EVAL_PREFIX}_${ckpt_tag}_contact_dist"
    temporal_dir="runs/eval/${EVAL_PREFIX}_${ckpt_tag}_temporal_coupling"

    echo
    echo "============================================================"
    echo "[eval:${ckpt_name}] qual_eval -> ${qual_dir}"
    echo "============================================================"
    python scripts/stage_b_generator/qual_eval.py \
      --config "$CFG" \
      --ckpt "$ckpt_path" \
      --output-dir "$qual_dir" \
      --num-clips "$NUM_CLIPS" \
      --seed "$SEED"

    echo
    echo "[eval:${ckpt_name}] contact distance -> ${dist_dir}"
    python scripts/stage_b_generator/measure_contact_distance.py \
      --input-dir "${qual_dir}/full" \
      --input-dir "${qual_dir}/text_only" \
      --input-dir "${qual_dir}/swap" \
      --input-dir "${gt_dir}/gt_original" \
      --input-dir "${gt_dir}/gt_roundtrip" \
      --output-dir "$dist_dir"

    echo
    echo "[eval:${ckpt_name}] temporal coupling -> ${temporal_dir}"
    python scripts/stage_b_generator/measure_temporal_coupling.py \
      --input-dir "${qual_dir}/full" \
      --output-dir "$temporal_dir" \
      --fps 20 \
      --coupling-threshold 0.5 \
      --moving-speed-threshold 0.15
  done
else
  echo "EVAL=0, skipping offline eval."
fi

if [[ "$DUMP_WANDB" == "1" ]]; then
  out_csv="runs/wandb_logs/wandb_history_genB_v13_target_trajectory.csv"
  echo
  echo "============================================================"
  echo "[wandb] ${RUN_NAME} -> ${out_csv}"
  echo "============================================================"
  python scripts/stage_a_predictor/dump_wandb_history.py \
    --name "$RUN_NAME" \
    --project "$WANDB_PROJECT" \
    --output "$out_csv" \
    --print-summary
else
  echo "DUMP_WANDB=0, skipping wandb history export."
fi

echo
echo "============================================================"
echo "Done. Sync these back for analysis:"
echo "============================================================"
echo "  ${RUN_DIR}/{best_contact.pt,best_val.pt,final.pt,train_*.log}"
echo "  runs/eval/${EVAL_PREFIX}_gt_roundtrip_${NUM_CLIPS}/"
for ckpt_name in $CKPTS; do
  case "$ckpt_name" in
    best_contact) ckpt_tag="bc" ;;
    best_val) ckpt_tag="bv" ;;
    final) ckpt_tag="final" ;;
    *) ckpt_tag="$ckpt_name" ;;
  esac
  echo "  runs/eval/${EVAL_PREFIX}_${ckpt_tag}_qual/summary.json"
  echo "  runs/eval/${EVAL_PREFIX}_${ckpt_tag}_contact_dist/summary.json"
  echo "  runs/eval/${EVAL_PREFIX}_${ckpt_tag}_temporal_coupling/summary.json"
done
echo "  runs/wandb_logs/wandb_history_genB_v13_target_trajectory.csv"
