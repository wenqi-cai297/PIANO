#!/usr/bin/env bash
# Re-run guided_alignment_to_gt_roundtrip for every v17 condition so that
# the new fields added in commit 2026-05-03 (weighted_local_error,
# weighted_target_error, soft_contact_temporal_iou_pm2) appear in the
# aggregate summary. Idempotent: overwrites existing summary.json.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-/c/Users/cwq29/miniforge3/envs/piano/python.exe}"
ROOT="${ROOT:-runs/eval}"

read -r -d '' CONDITIONS <<'EOF' || true
stageB_v0_17_per_step_v16bc bc
stageB_v0_17_v16bc_c_no_gumbel bc
stageB_v0_17_v16bc_stacked bc
stageB_v0_17_v16bc_per_step_iters20 bc
stageB_v0_17_v16bc_e20_no_gumbel bc
stageB_v0_17_v16bc_per_step_iters50 bc
stageB_v0_17_v16bc_f10_gumbel bc
stageB_v0_17_v16bc_f20_gumbel bc
stageB_v0_17_v16bc_g_b1 bc
stageB_v0_17_v16bc_g_b2 bc
stageB_v0_17_v16bc_g_b5 bc
stageB_v0_17_v16bc_g_b10 bc
stageB_v0_17_v16bc_g_b20 bc
stageB_v0_17_v16final_per_step_iters20 final
stageB_v0_17_v16final_per_step_iters50 final
stageB_v0_17h_v16bc_pm0_sc0 bc
stageB_v0_17h_v16bc_pm0_5 bc
stageB_v0_17h_v16bc_pm1_0 bc
stageB_v0_17h_v16bc_pm2_0 bc
stageB_v0_17h_v16bc_pm10_sc0_1 bc
stageB_v0_17h_v16bc_pm10_sc0_5 bc
stageB_v0_17h_v16bc_pm10_sc1_0 bc
EOF

while IFS=' ' read -r prefix ckpt; do
    [[ -z "$prefix" ]] && continue
    gen_dir="${ROOT}/${prefix}_${ckpt}_qual/full_guided"
    gt_dir="${ROOT}/${prefix}_gt_roundtrip_80/gt_roundtrip"
    out_dir="${ROOT}/${prefix}_${ckpt}_guided_alignment_to_gt_roundtrip"
    if [[ ! -d "$gen_dir" || ! -d "$gt_dir" ]]; then
        echo "[skip] ${prefix}_${ckpt}: missing generated or gt dir"
        continue
    fi
    echo "[realign] ${prefix}_${ckpt}"
    "$PYTHON" scripts/stage_b_generator/measure_contact_alignment.py \
        --generated-dir "$gen_dir" \
        --gt-dir "$gt_dir" \
        --output-dir "$out_dir" \
        --detail compact > /dev/null 2>&1
done <<< "$CONDITIONS"
echo "=== realign done ==="
