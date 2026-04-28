#!/usr/bin/env bash
# Stage B v0.12 decoded-contact weight sweep.
#
# Rationale:
#   v0.11 diagnostics showed decoded_contact_aux.weight=0.10 gives a small
#   decoded-contact gradient ratio (median ~3%, final ~4%). Instead of running
#   one guessed weight per iteration, train a bracket of weights with the same
#   C2b full-RVQ decoded-contact objective and the same diagnostics, then run
#   one shared offline eval pass across all finished checkpoints.
#
# Defaults:
#   WEIGHTS="0.20 0.30 0.50 0.80"
#   NUM_CLIPS=80   # matches training-time contact selector
#   CKPTS="best_contact best_val final"
#
# Useful overrides:
#   WEIGHTS="0.10 0.20 0.30 0.50 0.80" bash scripts/stage_b_generator/run_v12_contact_weight_sweep.sh
#   TRAIN=0 bash scripts/stage_b_generator/run_v12_contact_weight_sweep.sh   # eval/dump only
#   EVAL=0  bash scripts/stage_b_generator/run_v12_contact_weight_sweep.sh   # train only
#   FORCE_TRAIN=1 bash scripts/stage_b_generator/run_v12_contact_weight_sweep.sh

set -euo pipefail

BASE_CFG="${BASE_CFG:-configs/training/generator_v11_full_rvq_decoded_contact_aux_diagnostics.yaml}"
SWEEP_NAME="${SWEEP_NAME:-stageB_v12_decoded_contact_weight_sweep}"
SWEEP_DIR="${SWEEP_DIR:-runs/sweeps/${SWEEP_NAME}}"
CONFIG_DIR="${CONFIG_DIR:-${SWEEP_DIR}/configs}"
WEIGHTS="${WEIGHTS:-0.20 0.30 0.50 0.80}"
CKPTS="${CKPTS:-best_contact best_val final}"
NUM_CLIPS="${NUM_CLIPS:-80}"
SEED="${SEED:-42}"
TRAIN="${TRAIN:-1}"
EVAL="${EVAL:-1}"
DUMP_WANDB="${DUMP_WANDB:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-piano}"

if [[ ! -f "$BASE_CFG" ]]; then
  echo "ERROR: base config not found: $BASE_CFG" >&2
  exit 1
fi

mkdir -p "$CONFIG_DIR" "$SWEEP_DIR" runs/wandb_logs

weight_tag() {
  python - "$1" <<'PY'
import math
import sys

w = float(sys.argv[1])
if 0.0 < w < 1.0 and math.isclose(w * 10.0, round(w * 10.0), rel_tol=0.0, abs_tol=1e-8):
    print(f"w0{int(round(w * 10.0))}")
elif math.isclose(w, round(w), rel_tol=0.0, abs_tol=1e-8):
    print(f"w{int(round(w)):02d}")
else:
    print("w" + f"{w:g}".replace(".", "p"))
PY
}

make_config() {
  local weight="$1"
  local tag="$2"
  local cfg_path="${CONFIG_DIR}/generator_v12_decoded_contact_${tag}_diagnostics.yaml"

  python - "$BASE_CFG" "$cfg_path" "$weight" "$tag" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf

base_cfg = Path(sys.argv[1])
out_cfg = Path(sys.argv[2])
weight = float(sys.argv[3])
tag = sys.argv[4]

cfg = OmegaConf.load(base_cfg)
cfg.training.decoded_contact_aux.weight = weight
cfg.logging.run_name = f"predictor_stageB_v12_decoded_contact_{tag}_diagnostics"
cfg.output_dir = f"runs/training/generator_v12_decoded_contact_{tag}_diagnostics"

out_cfg.parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(cfg, out_cfg)
print(out_cfg)
print(cfg.logging.run_name)
print(cfg.output_dir)
PY
}

read -r -a WEIGHT_ARRAY <<< "$WEIGHTS"
read -r -a CKPT_ARRAY <<< "$CKPTS"

declare -a TAGS=()
declare -a CFGS=()
declare -a RUN_DIRS=()
declare -a RUN_NAMES=()

echo
echo "============================================================"
echo "Preparing configs for ${SWEEP_NAME}"
echo "  weights: ${WEIGHTS}"
echo "  base cfg: ${BASE_CFG}"
echo "============================================================"

for weight in "${WEIGHT_ARRAY[@]}"; do
  tag="$(weight_tag "$weight")"
  mapfile -t cfg_info < <(make_config "$weight" "$tag")
  cfg_path="${cfg_info[0]}"
  run_name="${cfg_info[1]}"
  run_dir="${cfg_info[2]}"

  TAGS+=("$tag")
  CFGS+=("$cfg_path")
  RUN_NAMES+=("$run_name")
  RUN_DIRS+=("$run_dir")

  echo "  ${tag}: weight=${weight} cfg=${cfg_path} run_dir=${run_dir}"
done

if [[ "$TRAIN" == "1" ]]; then
  for i in "${!TAGS[@]}"; do
    tag="${TAGS[$i]}"
    cfg_path="${CFGS[$i]}"
    run_dir="${RUN_DIRS[$i]}"

    mkdir -p "$run_dir"
    if [[ -f "${run_dir}/final.pt" && "$FORCE_TRAIN" != "1" ]]; then
      echo
      echo "============================================================"
      echo "[train:${tag}] final.pt exists; skipping training"
      echo "  set FORCE_TRAIN=1 to overwrite/retrain"
      echo "============================================================"
      continue
    fi

    echo
    echo "============================================================"
    echo "[train:${tag}] ${cfg_path}"
    echo "============================================================"
    accelerate launch --config_file configs/accelerate_config.yaml \
      -m piano.training.train_generator \
      --config "$cfg_path" \
      2>&1 | tee "${run_dir}/train_$(date +%Y%m%d_%H%M%S).log"
  done
else
  echo
  echo "TRAIN=0, skipping training."
fi

if [[ "$EVAL" == "1" ]]; then
  gt_dir="runs/eval/stageB_v0_12_sweep_gt_roundtrip_${NUM_CLIPS}"
  if [[ ! -d "${gt_dir}/gt_original" || ! -d "${gt_dir}/gt_roundtrip" ]]; then
    echo
    echo "============================================================"
    echo "[eval] GT VQ roundtrip refs -> ${gt_dir}"
    echo "============================================================"
    python scripts/stage_b_generator/gt_vq_roundtrip.py \
      --config "${CFGS[0]}" \
      --num-clips "$NUM_CLIPS" \
      --seed "$SEED" \
      --output-dir "$gt_dir"
  else
    echo
    echo "[eval] Reusing GT VQ roundtrip refs: ${gt_dir}"
  fi

  for i in "${!TAGS[@]}"; do
    tag="${TAGS[$i]}"
    cfg_path="${CFGS[$i]}"
    run_dir="${RUN_DIRS[$i]}"

    for ckpt_name in "${CKPT_ARRAY[@]}"; do
      ckpt_path="${run_dir}/${ckpt_name}.pt"
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

      qual_dir="runs/eval/stageB_v0_12_${tag}_${ckpt_tag}_qual"
      dist_dir="runs/eval/stageB_v0_12_${tag}_${ckpt_tag}_contact_dist"

      echo
      echo "============================================================"
      echo "[eval:${tag}:${ckpt_name}] qual_eval -> ${qual_dir}"
      echo "============================================================"
      python scripts/stage_b_generator/qual_eval.py \
        --config "$cfg_path" \
        --ckpt "$ckpt_path" \
        --output-dir "$qual_dir" \
        --num-clips "$NUM_CLIPS" \
        --seed "$SEED"

      echo
      echo "[eval:${tag}:${ckpt_name}] contact distance -> ${dist_dir}"
      python scripts/stage_b_generator/measure_contact_distance.py \
        --input-dir "${qual_dir}/full" \
        --input-dir "${qual_dir}/text_only" \
        --input-dir "${qual_dir}/swap" \
        --input-dir "${gt_dir}/gt_original" \
        --input-dir "${gt_dir}/gt_roundtrip" \
        --output-dir "$dist_dir"
    done
  done
else
  echo
  echo "EVAL=0, skipping offline eval."
fi

if [[ "$DUMP_WANDB" == "1" ]]; then
  for i in "${!TAGS[@]}"; do
    tag="${TAGS[$i]}"
    run_name="${RUN_NAMES[$i]}"
    out_csv="runs/wandb_logs/wandb_history_genB_v12_${tag}.csv"

    echo
    echo "============================================================"
    echo "[wandb:${tag}] ${run_name} -> ${out_csv}"
    echo "============================================================"
    python scripts/stage_a_predictor/dump_wandb_history.py \
      --name "$run_name" \
      --project "$WANDB_PROJECT" \
      --output "$out_csv" \
      --print-summary
  done
else
  echo
  echo "DUMP_WANDB=0, skipping wandb history export."
fi

echo
echo "============================================================"
echo "Sweep complete. Sync these back for analysis:"
echo "============================================================"
echo "  ${CONFIG_DIR}/"
echo "  runs/eval/stageB_v0_12_sweep_gt_roundtrip_${NUM_CLIPS}/"
for tag in "${TAGS[@]}"; do
  echo "  runs/eval/stageB_v0_12_${tag}_*_qual/summary.json"
  echo "  runs/eval/stageB_v0_12_${tag}_*_contact_dist/summary.json"
  echo "  runs/wandb_logs/wandb_history_genB_v12_${tag}.csv"
done
