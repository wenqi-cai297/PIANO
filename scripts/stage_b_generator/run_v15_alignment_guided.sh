#!/usr/bin/env bash
# Train/evaluate Stage B v0.15 alignment-aware contact loss plus full-RVQ guidance.
#
# Useful overrides:
#   TRAIN=0 bash scripts/stage_b_generator/run_v15_alignment_guided.sh
#   CKPTS="best_contact" GUIDANCE_STEPS=60 bash scripts/stage_b_generator/run_v15_alignment_guided.sh
#   SUMMARY_DETAIL=full bash scripts/stage_b_generator/run_v15_alignment_guided.sh

set -euo pipefail

export CFG="${CFG:-configs/training/generator_v15_alignment_guided.yaml}"
export RUN_DIR="${RUN_DIR:-runs/training/generator_v15_alignment_guided}"
export RUN_NAME="${RUN_NAME:-predictor_stageB_v15_alignment_guided}"
export EVAL_PREFIX="${EVAL_PREFIX:-stageB_v0_15_alignment_guided}"
export WANDB_OUTPUT="${WANDB_OUTPUT:-runs/wandb_logs/wandb_history_genB_v15_alignment_guided.csv}"
export WANDB_COLUMNS="${WANDB_COLUMNS:-epoch,loss,loss_base,loss_residual,loss_decoded_contact,loss_weighted_decoded_contact,acc,acc_residual,decoded_contact_aux_target_position,decoded_contact_aux_target_velocity,decoded_contact_aux_part_margin,decoded_contact_aux_part_margin_active_frac,decoded_contact_aux_segment_consistency,decoded_contact_aux_mean_min_dist,decoded_contact_aux_hard_forward,gamma_int_abs_mean,gamma_int_res_abs_mean,val_loss,val_loss_base,val_loss_residual,val_loss_decoded_contact,val_loss_weighted_decoded_contact,val_acc,val_acc_residual,val_decoded_contact_aux_target_position,val_decoded_contact_aux_target_velocity,val_decoded_contact_aux_part_margin,val_decoded_contact_aux_part_margin_active_frac,val_decoded_contact_aux_segment_consistency,val_decoded_contact_aux_mean_min_dist,val_decoded_contact_aux_hard_forward,contact_alignment_contact_score,contact_alignment_primary_error,contact_alignment_moving_target_error,contact_alignment_moving_same_part_recall,contact_alignment_same_part_recall,contact_composite_contact_score,contact_mean_min_dist,contact_moving_close_frame_frac,contact_moving_coupled_frame_frac,contact_moving_close_but_uncoupled_frac,contact_n_clips,lr,epoch_time_sec}"

# Full-RVQ, target-alignment guidance is deliberately enabled for offline eval.
# Use GUIDANCE_STEPS=0 if you only want the raw generator distribution.
export GUIDANCE_STEPS="${GUIDANCE_STEPS:-30}"
export GUIDANCE_LAYERS="${GUIDANCE_LAYERS:-full_rvq}"
export GUIDANCE_LOSS="${GUIDANCE_LOSS:-target}"
export GUIDANCE_LR="${GUIDANCE_LR:-6e-2}"
export GUIDANCE_INIT_SCALE="${GUIDANCE_INIT_SCALE:-3.0}"
export GUIDANCE_RESIDUAL_SEED="${GUIDANCE_RESIDUAL_SEED:-42}"

exec bash scripts/stage_b_generator/run_v13_target_trajectory.sh
