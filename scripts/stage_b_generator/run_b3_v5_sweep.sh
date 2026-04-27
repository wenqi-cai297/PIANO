#!/usr/bin/env bash
# B3 v5 dual-flag sweep — disambiguate RNG drift vs autoregressive feedback
# in the residual transformer rerun.
#
# Background:
#   v4 (commit 6e443a0, metric-mode loss) landed Branch C of the decision
#   tree: per-clip mixed (largebox -14 cm, plasticbox_037 +16 cm). Smoking
#   gun was plasticbox_014 with 0/23 base flips yet +7.3 cm contact
#   regression — the only post-argmax delta is the unconditional
#   res_transformer.generate rerun. Source-read of MoMask transformer.py:949
#   + tools.py:90-95 identified two stacked mechanisms inside
#   ResidualTransformer.generate:
#     1. RNG drift via gumbel_sample on global default RNG (no manual_seed).
#     2. Autoregressive feedback (layer-i sample → layer-i+1 logits via
#        history_sum).
#
# This script runs 4 combos × 5 clips on v0.6 b1_bestval to disambiguate:
#   baseline  — no flag (sanity check, should match v4 result 27.6 cm)
#   seed      — --guidance-residual-seed 42 (eliminates 1 only)
#   norerun   — --guidance-no-residual-rerun (eliminates both)
#   both      — both flags (control)
#
# Wallclock: ~5 min/combo × 4 = ~20 min total on 2× A6000.
#
# Outputs (under runs/eval/):
#   stageB_v0_6_b1_v5_{baseline,seed,norerun,both}/
#     full/generated.npz, full_guided/generated.npz + guidance_trace.json,
#     text_only/, swap/, summary.json
#   stageB_v0_6_b1_v5_{baseline,seed,norerun,both}_contact_dist/summary.json
#
# Decision matrix interpreting the 4 results: see
# analyses/2026-04-28_b1_b3_iteration_log.md §"v5 — Dual-flag diagnostic".

set -euo pipefail

CKPT="runs/training/generator_v06_per_head_gamma/best_val.pt"
CFG="configs/training/generator_v06_per_head_gamma.yaml"
SEED=42
STEPS=30

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: ckpt not found at $CKPT — run from PIANO repo root." >&2
  exit 1
fi

for combo in baseline seed norerun both; do
  case "$combo" in
    baseline) flags=() ;;
    seed)     flags=(--guidance-residual-seed "$SEED") ;;
    norerun)  flags=(--guidance-no-residual-rerun) ;;
    both)     flags=(--guidance-residual-seed "$SEED" --guidance-no-residual-rerun) ;;
  esac

  out_qual="runs/eval/stageB_v0_6_b1_v5_${combo}"
  out_dist="runs/eval/stageB_v0_6_b1_v5_${combo}_contact_dist"

  echo
  echo "============================================================"
  echo "[${combo}] qual_eval -> ${out_qual}"
  echo "  flags: ${flags[*]:-<none>}"
  echo "============================================================"
  python scripts/stage_b_generator/qual_eval.py \
    --config "$CFG" \
    --ckpt "$CKPT" \
    --guidance-steps "$STEPS" \
    --output-dir "$out_qual" \
    "${flags[@]}"

  echo
  echo "[${combo}] measure_contact_distance -> ${out_dist}"
  python scripts/stage_b_generator/measure_contact_distance.py \
    --input-dir "${out_qual}/full" \
    --input-dir "${out_qual}/full_guided" \
    --input-dir "${out_qual}/text_only" \
    --input-dir "${out_qual}/swap" \
    --output-dir "$out_dist"
done

echo
echo "============================================================"
echo "All 4 combos done. Sync these back to local for analysis:"
echo "============================================================"
for combo in baseline seed norerun both; do
  echo "  runs/eval/stageB_v0_6_b1_v5_${combo}/full_guided/guidance_trace.json"
  echo "  runs/eval/stageB_v0_6_b1_v5_${combo}_contact_dist/summary.json"
done
echo
echo "Decision matrix:"
echo "  analyses/2026-04-28_b1_b3_iteration_log.md §'v5 — Dual-flag diagnostic'"
