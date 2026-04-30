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
WANDB_OUTPUT="${WANDB_OUTPUT:-runs/wandb_logs/wandb_history_genB_v13_target_trajectory.csv}"
SUMMARY_DETAIL="${SUMMARY_DETAIL:-compact}"
GUIDANCE_STEPS="${GUIDANCE_STEPS:-0}"
GUIDANCE_LAYERS="${GUIDANCE_LAYERS:-base}"
GUIDANCE_LOSS="${GUIDANCE_LOSS:-metric}"
GUIDANCE_LR="${GUIDANCE_LR:-6e-2}"
GUIDANCE_INIT_SCALE="${GUIDANCE_INIT_SCALE:-3.0}"
GUIDANCE_RESIDUAL_SEED="${GUIDANCE_RESIDUAL_SEED:-}"
GUIDANCE_NO_RESIDUAL_RERUN="${GUIDANCE_NO_RESIDUAL_RERUN:-0}"
PER_STEP_ITERS="${PER_STEP_ITERS:-0}"
PER_STEP_LR="${PER_STEP_LR:-6e-2}"
PER_STEP_TEMPERATURE="${PER_STEP_TEMPERATURE:-1.0}"
PER_STEP_START_STEP="${PER_STEP_START_STEP:-0}"
PER_STEP_GUMBEL_SCALE="${PER_STEP_GUMBEL_SCALE:-0.0}"
# v17-H (B2): part_margin + segment_consistency in per-step inner loss.
# 0.0 (default) = back-compat (just primary masked L2). See
# analyses/2026-05-01_v17_re_diagnosis.md §B2 for the recommended sweep.
PER_STEP_PART_MARGIN_WEIGHT="${PER_STEP_PART_MARGIN_WEIGHT:-0.0}"
PER_STEP_PART_MARGIN_M="${PER_STEP_PART_MARGIN_M:-0.08}"
PER_STEP_SEGMENT_CONSISTENCY_WEIGHT="${PER_STEP_SEGMENT_CONSISTENCY_WEIGHT:-0.0}"
GAMMA_INT_BOOST="${GAMMA_INT_BOOST:-1.0}"
WANDB_COLUMNS="${WANDB_COLUMNS:-epoch,loss,loss_base,loss_residual,loss_decoded_contact,loss_weighted_decoded_contact,acc,acc_residual,decoded_contact_aux_target_position,decoded_contact_aux_target_velocity,decoded_contact_aux_part_margin,decoded_contact_aux_part_margin_active_frac,decoded_contact_aux_segment_consistency,decoded_contact_aux_mean_min_dist,gamma_int_abs_mean,gamma_int_res_abs_mean,val_loss,val_loss_base,val_loss_residual,val_loss_decoded_contact,val_loss_weighted_decoded_contact,val_acc,val_acc_residual,val_decoded_contact_aux_target_position,val_decoded_contact_aux_target_velocity,val_decoded_contact_aux_part_margin,val_decoded_contact_aux_part_margin_active_frac,val_decoded_contact_aux_segment_consistency,val_decoded_contact_aux_mean_min_dist,contact_alignment_contact_score,contact_alignment_primary_error,contact_alignment_moving_target_error,contact_alignment_moving_same_part_recall,contact_composite_contact_score,contact_mean_min_dist,contact_moving_close_frame_frac,contact_moving_coupled_frame_frac,contact_moving_close_but_uncoupled_frac,contact_n_clips,lr,epoch_time_sec}"

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
    guided_temporal_dir="runs/eval/${EVAL_PREFIX}_${ckpt_tag}_guided_temporal_coupling"
    alignment_dir="runs/eval/${EVAL_PREFIX}_${ckpt_tag}_alignment_to_gt_roundtrip"
    guided_alignment_dir="runs/eval/${EVAL_PREFIX}_${ckpt_tag}_guided_alignment_to_gt_roundtrip"

    echo
    echo "============================================================"
    echo "[eval:${ckpt_name}] qual_eval -> ${qual_dir}"
    echo "============================================================"
    qual_cmd=(
      python scripts/stage_b_generator/qual_eval.py
      --config "$CFG" \
      --ckpt "$ckpt_path" \
      --output-dir "$qual_dir" \
      --num-clips "$NUM_CLIPS" \
      --seed "$SEED" \
      --summary-detail "$SUMMARY_DETAIL"
    )
    if [[ "$GUIDANCE_STEPS" != "0" || "$PER_STEP_ITERS" != "0" ]]; then
      qual_cmd+=(
        --guidance-steps "$GUIDANCE_STEPS"
        --guidance-layers "$GUIDANCE_LAYERS"
        --guidance-loss "$GUIDANCE_LOSS"
        --guidance-lr "$GUIDANCE_LR"
        --guidance-init-scale "$GUIDANCE_INIT_SCALE"
      )
      if [[ -n "$GUIDANCE_RESIDUAL_SEED" ]]; then
        qual_cmd+=(--guidance-residual-seed "$GUIDANCE_RESIDUAL_SEED")
      fi
      if [[ "$GUIDANCE_NO_RESIDUAL_RERUN" == "1" ]]; then
        qual_cmd+=(--guidance-no-residual-rerun)
      fi
      if [[ "$PER_STEP_ITERS" != "0" ]]; then
        qual_cmd+=(
          --per-step-iters "$PER_STEP_ITERS"
          --per-step-lr "$PER_STEP_LR"
          --per-step-temperature "$PER_STEP_TEMPERATURE"
          --per-step-start-step "$PER_STEP_START_STEP"
          --per-step-gumbel-scale "$PER_STEP_GUMBEL_SCALE"
          --per-step-part-margin-weight "$PER_STEP_PART_MARGIN_WEIGHT"
          --per-step-part-margin-m "$PER_STEP_PART_MARGIN_M"
          --per-step-segment-consistency-weight "$PER_STEP_SEGMENT_CONSISTENCY_WEIGHT"
        )
      fi
    fi
    if [[ "$GAMMA_INT_BOOST" != "1.0" ]]; then
      qual_cmd+=(--gamma-int-boost "$GAMMA_INT_BOOST")
    fi
    "${qual_cmd[@]}"

    echo
    echo "[eval:${ckpt_name}] contact distance -> ${dist_dir}"
    dist_cmd=(
      python scripts/stage_b_generator/measure_contact_distance.py
      --input-dir "${qual_dir}/full"
      --input-dir "${qual_dir}/text_only"
      --input-dir "${qual_dir}/swap"
    )
    if [[ -d "${qual_dir}/full_guided" ]]; then
      dist_cmd+=(--input-dir "${qual_dir}/full_guided")
    fi
    dist_cmd+=(
      --input-dir "${gt_dir}/gt_original"
      --input-dir "${gt_dir}/gt_roundtrip"
      --output-dir "$dist_dir"
      --detail "$SUMMARY_DETAIL"
    )
    "${dist_cmd[@]}"

    echo
    echo "[eval:${ckpt_name}] temporal coupling -> ${temporal_dir}"
    python scripts/stage_b_generator/measure_temporal_coupling.py \
      --input-dir "${qual_dir}/full" \
      --output-dir "$temporal_dir" \
      --fps 20 \
      --coupling-threshold 0.5 \
      --moving-speed-threshold 0.15 \
      --detail "$SUMMARY_DETAIL"

    if [[ -d "${qual_dir}/full_guided" ]]; then
      echo
      echo "[eval:${ckpt_name}] guided temporal coupling -> ${guided_temporal_dir}"
      python scripts/stage_b_generator/measure_temporal_coupling.py \
        --input-dir "${qual_dir}/full_guided" \
        --output-dir "$guided_temporal_dir" \
        --fps 20 \
        --coupling-threshold 0.5 \
        --moving-speed-threshold 0.15 \
        --detail "$SUMMARY_DETAIL"
    fi

    echo
    echo "[eval:${ckpt_name}] alignment to GT roundtrip -> ${alignment_dir}"
    python scripts/stage_b_generator/measure_contact_alignment.py \
      --generated-dir "${qual_dir}/full" \
      --gt-dir "${gt_dir}/gt_roundtrip" \
      --output-dir "$alignment_dir" \
      --fps 20 \
      --detail "$SUMMARY_DETAIL"

    if [[ -d "${qual_dir}/full_guided" ]]; then
      echo
      echo "[eval:${ckpt_name}] guided alignment to GT roundtrip -> ${guided_alignment_dir}"
      python scripts/stage_b_generator/measure_contact_alignment.py \
        --generated-dir "${qual_dir}/full_guided" \
        --gt-dir "${gt_dir}/gt_roundtrip" \
        --output-dir "$guided_alignment_dir" \
        --fps 20 \
        --detail "$SUMMARY_DETAIL"
    fi
  done
else
  echo "EVAL=0, skipping offline eval."
fi

if [[ "$DUMP_WANDB" == "1" ]]; then
  out_csv="$WANDB_OUTPUT"
  echo
  echo "============================================================"
  echo "[wandb] ${RUN_NAME} -> ${out_csv}"
  echo "============================================================"
  wandb_cmd=(
    python scripts/stage_a_predictor/dump_wandb_history.py
    --name "$RUN_NAME"
    --project "$WANDB_PROJECT"
    --output "$out_csv"
    --print-summary
  )
  if [[ -n "$WANDB_COLUMNS" ]]; then
    wandb_cmd+=(--columns "$WANDB_COLUMNS")
  fi
  "${wandb_cmd[@]}"
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
  echo "  runs/eval/${EVAL_PREFIX}_${ckpt_tag}_alignment_to_gt_roundtrip/summary.json"
  if [[ "$GUIDANCE_STEPS" != "0" || "$PER_STEP_ITERS" != "0" ]]; then
    echo "  runs/eval/${EVAL_PREFIX}_${ckpt_tag}_qual/full_guided/{summary.json,guidance_trace.json,generated.npz}"
    echo "  runs/eval/${EVAL_PREFIX}_${ckpt_tag}_guided_temporal_coupling/summary.json"
    echo "  runs/eval/${EVAL_PREFIX}_${ckpt_tag}_guided_alignment_to_gt_roundtrip/summary.json"
  fi
done
echo "  ${WANDB_OUTPUT}"
