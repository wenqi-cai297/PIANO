#!/usr/bin/env bash
# Stage B v0.9 (C2 decoded contact aux loss) — full eval pipeline.
#
# Run AFTER training finishes:
#   accelerate launch --config_file configs/accelerate_config.yaml \
#     -m piano.training.train_generator \
#     --config configs/training/generator_v09_decoded_contact_aux.yaml
#
# This script generates (1) qual eval JSON+npz on the B1 best_contact.pt
# checkpoint with 20 stratified clips, (2) the contact-distance summary
# vs GT references, (3) the same on best_val.pt for CE-vs-contact
# comparison, and (4) the wandb history CSV.
#
# Reference baselines for comparison (from prior runs):
#   v0.6 b1_bestval `full`        = 23.6 cm   (2026-04-28 B1 retrain on canonical 5)
#   v0.8 (C1) `full`              = 43.62 cm  (analyses/2026-04-28_c1_c2_landing_review.md, 5-clip)
#   v0.8 (C1) `text_only`         = 44.01 cm  (5-clip)
#   GT original                   = 10.52 cm
#   GT roundtrip (all-RVQ)        = 11.29 cm
#
# Decision rule on v0.9 `full` agg vs v0.8 `full` 43.62 cm + text_only gap:
#   - drop ≥ 5 cm AND open meaningful full-vs-text gap → C2 working,
#     queue weight/temperature/num_object_points sweep.
#   - drop < 1 cm OR no full-vs-text gap → C2 ineffective; pivot to C2b
#     (route decoded loss through residual logits, not GT residual
#     teacher forcing). See analyses/2026-04-28_c1_c2_landing_review.md
#     §"Next Action" + §"Known Assumptions And Risks".
#
# The 20-clip stratified eval is wider than the v0.8 5-clip canonical;
# direct comparison requires translating the v0.8 number to the same set.
# Cheap sanity-check: also run on the 5 canonical clips by passing
# --num-clips 5 manually for an apples-to-apples vs v0.8.

set -euo pipefail

CFG="configs/training/generator_v09_decoded_contact_aux.yaml"
RUN_DIR="runs/training/generator_v09_decoded_contact_aux"
WANDB_RUN_NAME="predictor_stageB_v09_decoded_contact_aux"
NUM_CLIPS=20
SEED=42

# Reference GT directories (precomputed earlier; reuse for d_codebook /
# d_total decomposition).
GT_REF_DIR="runs/eval/stageB_v0_4_gt_roundtrip"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: training run dir not found: $RUN_DIR" >&2
  echo "       did v0.9 training finish?" >&2
  exit 1
fi
for ckpt_name in best_contact.pt best_val.pt; do
  if [[ ! -f "$RUN_DIR/$ckpt_name" ]]; then
    echo "ERROR: ckpt $RUN_DIR/$ckpt_name not found" >&2
    exit 1
  fi
done
if [[ ! -d "$GT_REF_DIR/gt_original" || ! -d "$GT_REF_DIR/gt_roundtrip" ]]; then
  echo "WARN: GT roundtrip refs missing at $GT_REF_DIR/{gt_original,gt_roundtrip}" >&2
  echo "      contact_dist will skip those baselines." >&2
fi

# ----------------------------------------------------------------------
# 1. Qual eval on best_contact.pt (the B1 ckpt that selects on contact)
# ----------------------------------------------------------------------
echo
echo "============================================================"
echo "[1/4] qual_eval on best_contact.pt (${NUM_CLIPS} stratified clips)"
echo "============================================================"
QUAL_BC="runs/eval/stageB_v0_9_bc_qual"
python scripts/stage_b_generator/qual_eval.py \
  --config "$CFG" \
  --ckpt "$RUN_DIR/best_contact.pt" \
  --output-dir "$QUAL_BC" \
  --num-clips "$NUM_CLIPS" \
  --seed "$SEED"

echo
echo "============================================================"
echo "[2/4] measure_contact_distance on best_contact.pt outputs"
echo "============================================================"
DIST_BC="runs/eval/stageB_v0_9_bc_contact_dist"
DIST_INPUT_ARGS=(
  --input-dir "${QUAL_BC}/full"
  --input-dir "${QUAL_BC}/text_only"
  --input-dir "${QUAL_BC}/swap"
)
if [[ -d "$GT_REF_DIR/gt_original" ]]; then
  DIST_INPUT_ARGS+=(--input-dir "${GT_REF_DIR}/gt_original")
fi
if [[ -d "$GT_REF_DIR/gt_roundtrip" ]]; then
  DIST_INPUT_ARGS+=(--input-dir "${GT_REF_DIR}/gt_roundtrip")
fi
python scripts/stage_b_generator/measure_contact_distance.py \
  "${DIST_INPUT_ARGS[@]}" \
  --output-dir "$DIST_BC"

# ----------------------------------------------------------------------
# 3. Qual eval on best_val.pt (CE-best ckpt) for CE-vs-contact comparison
#    on the same stratified set. Cheap: ~10 min; same generator weights
#    as best_contact.pt only differ by which epoch the selector picked.
# ----------------------------------------------------------------------
echo
echo "============================================================"
echo "[3/4] qual_eval + contact_dist on best_val.pt (CE-best ckpt)"
echo "============================================================"
QUAL_BV="runs/eval/stageB_v0_9_bv_qual"
DIST_BV="runs/eval/stageB_v0_9_bv_contact_dist"
python scripts/stage_b_generator/qual_eval.py \
  --config "$CFG" \
  --ckpt "$RUN_DIR/best_val.pt" \
  --output-dir "$QUAL_BV" \
  --num-clips "$NUM_CLIPS" \
  --seed "$SEED"

DIST_BV_ARGS=(
  --input-dir "${QUAL_BV}/full"
  --input-dir "${QUAL_BV}/text_only"
  --input-dir "${QUAL_BV}/swap"
)
if [[ -d "$GT_REF_DIR/gt_original" ]]; then
  DIST_BV_ARGS+=(--input-dir "${GT_REF_DIR}/gt_original")
fi
if [[ -d "$GT_REF_DIR/gt_roundtrip" ]]; then
  DIST_BV_ARGS+=(--input-dir "${GT_REF_DIR}/gt_roundtrip")
fi
python scripts/stage_b_generator/measure_contact_distance.py \
  "${DIST_BV_ARGS[@]}" \
  --output-dir "$DIST_BV"

# ----------------------------------------------------------------------
# 4. Wandb history CSV (--name uses display_name from cfg.logging.run_name)
# ----------------------------------------------------------------------
echo
echo "============================================================"
echo "[4/4] dump_wandb_history -> runs/wandb_logs/wandb_history_genB_v09.csv"
echo "============================================================"
mkdir -p runs/wandb_logs
python scripts/stage_a_predictor/dump_wandb_history.py \
  --name "$WANDB_RUN_NAME" \
  --output runs/wandb_logs/wandb_history_genB_v09.csv

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
echo
echo "============================================================"
echo "v0.9 eval done. Sync these back to local for analysis:"
echo "============================================================"
echo "  ${DIST_BC}/summary.json    # primary: best_contact.pt contact"
echo "  ${QUAL_BC}/summary.json    # token Hamming + RMS (best_contact.pt)"
echo "  ${DIST_BV}/summary.json    # secondary: best_val.pt contact"
echo "  ${QUAL_BV}/summary.json"
echo "  runs/wandb_logs/wandb_history_genB_v09.csv"
echo
echo "Compare full agg vs v0.8 (43.62 cm) and text_only gap:"
echo "  python -c \"import json; "
echo "    d = json.load(open('${DIST_BC}/summary.json')); "
echo "    for k, v in d['conditions'].items(): "
echo "      if 'agg' in v: print(f'  {k}: {v[\\\"agg\\\"][\\\"agg_mean_min_dist_per_frame\\\"]*100:.2f} cm')\""
