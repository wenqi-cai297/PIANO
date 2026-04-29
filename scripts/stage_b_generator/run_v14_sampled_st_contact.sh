#!/usr/bin/env bash
# Train/evaluate Stage B v0.14 sampled straight-through contact loss.
#
# This is a thin wrapper around the v13 train/eval runner with v14 defaults.
# Useful overrides:
#   TRAIN=0 bash scripts/stage_b_generator/run_v14_sampled_st_contact.sh
#   CKPTS="best_contact final" bash scripts/stage_b_generator/run_v14_sampled_st_contact.sh
#   SUMMARY_DETAIL=full bash scripts/stage_b_generator/run_v14_sampled_st_contact.sh

set -euo pipefail

export CFG="${CFG:-configs/training/generator_v14_sampled_st_contact.yaml}"
export RUN_DIR="${RUN_DIR:-runs/training/generator_v14_sampled_st_contact}"
export RUN_NAME="${RUN_NAME:-predictor_stageB_v14_sampled_st_contact}"
export EVAL_PREFIX="${EVAL_PREFIX:-stageB_v0_14_sampled_st_contact}"
export WANDB_OUTPUT="${WANDB_OUTPUT:-runs/wandb_logs/wandb_history_genB_v14_sampled_st_contact.csv}"
export WANDB_COLUMNS="${WANDB_COLUMNS:-epoch,loss,loss_base,loss_residual,loss_decoded_contact,loss_weighted_decoded_contact,acc,acc_residual,decoded_contact_aux_target_position,decoded_contact_aux_target_velocity,decoded_contact_aux_mean_min_dist,decoded_contact_aux_hard_forward,gamma_int_abs_mean,gamma_int_res_abs_mean,val_loss,val_loss_base,val_loss_residual,val_loss_decoded_contact,val_loss_weighted_decoded_contact,val_acc,val_acc_residual,val_decoded_contact_aux_target_position,val_decoded_contact_aux_target_velocity,val_decoded_contact_aux_mean_min_dist,val_decoded_contact_aux_hard_forward,contact_composite_contact_score,contact_mean_min_dist,contact_moving_close_frame_frac,contact_moving_coupled_frame_frac,contact_moving_close_but_uncoupled_frac,contact_n_clips,lr,epoch_time_sec}"

exec bash scripts/stage_b_generator/run_v13_target_trajectory.sh
