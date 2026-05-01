#!/usr/bin/env bash
# Run penetration (N1/N2) + motion-quality jerk (N7) on every v17 condition's
# generated motion plus GT_orig / GT_roundtrip references. Output JSON files
# go alongside the existing alignment / contact_dist / temporal_coupling
# summaries so the unified summarize script can consume everything from one
# tree.
#
# This script is idempotent — skips conditions whose new metric output
# already exists. Total wall-clock for a fresh 20-condition sweep on a
# laptop CPU is ~15 min.

set -euo pipefail

cd "$(dirname "$0")/../.."

EVAL_ROOT="${EVAL_ROOT:-runs/eval}"
PEN_OUT="${PEN_OUT:-${EVAL_ROOT}/_unified_metrics/penetration}"
JERK_OUT="${JERK_OUT:-${EVAL_ROOT}/_unified_metrics/jerk}"
mkdir -p "${PEN_OUT}" "${JERK_OUT}"

# ============================================================================
# Conditions to evaluate. Each line: <label>|<input_dir>
# Labels are short and used in output JSON filenames.
# ============================================================================
read -r -d '' CONDITIONS <<'EOF' || true
gt_orig|stageB_v0_17_v16final_per_step_iters20_gt_roundtrip_80/gt_original
gt_roundtrip|stageB_v0_17_v16final_per_step_iters20_gt_roundtrip_80/gt_roundtrip
v17C_v16bc|stageB_v0_17_per_step_v16bc_bc_qual/full_guided
v17C_v16bc_no_gumbel|stageB_v0_17_v16bc_c_no_gumbel_bc_qual/full_guided
v17D_stacked|stageB_v0_17_v16bc_stacked_bc_qual/full_guided
v17E20_v16bc|stageB_v0_17_v16bc_per_step_iters20_bc_qual/full_guided
v17E20_v16bc_no_gumbel|stageB_v0_17_v16bc_e20_no_gumbel_bc_qual/full_guided
v17E50_v16bc|stageB_v0_17_v16bc_per_step_iters50_bc_qual/full_guided
v17F10_gumbel|stageB_v0_17_v16bc_f10_gumbel_bc_qual/full_guided
v17F20_gumbel|stageB_v0_17_v16bc_f20_gumbel_bc_qual/full_guided
v17G_b1|stageB_v0_17_v16bc_g_b1_bc_qual/full_guided
v17G_b2|stageB_v0_17_v16bc_g_b2_bc_qual/full_guided
v17G_b5|stageB_v0_17_v16bc_g_b5_bc_qual/full_guided
v17G_b10|stageB_v0_17_v16bc_g_b10_bc_qual/full_guided
v17G_b20|stageB_v0_17_v16bc_g_b20_bc_qual/full_guided
B1_v17E20_final|stageB_v0_17_v16final_per_step_iters20_final_qual/full_guided
B1_v17E50_final|stageB_v0_17_v16final_per_step_iters50_final_qual/full_guided
B2_pm0_sc0|stageB_v0_17h_v16bc_pm0_sc0_bc_qual/full_guided
B2_pm0_5|stageB_v0_17h_v16bc_pm0_5_bc_qual/full_guided
B2_pm1_0|stageB_v0_17h_v16bc_pm1_0_bc_qual/full_guided
B2_pm2_0|stageB_v0_17h_v16bc_pm2_0_bc_qual/full_guided
B2_pm10_sc0_1|stageB_v0_17h_v16bc_pm10_sc0_1_bc_qual/full_guided
B2_pm10_sc0_5|stageB_v0_17h_v16bc_pm10_sc0_5_bc_qual/full_guided
B2_pm10_sc1_0|stageB_v0_17h_v16bc_pm10_sc1_0_bc_qual/full_guided
EOF

run_metric_for_condition() {
    local label="$1"
    local input_dir="$2"
    local out_pen="${PEN_OUT}/${label}_summary.json"
    local out_jerk="${JERK_OUT}/${label}_summary.json"

    if [[ ! -f "${EVAL_ROOT}/${input_dir}/generated.npz" ]]; then
        echo "  [skip ${label}] missing ${EVAL_ROOT}/${input_dir}/generated.npz"
        return
    fi

    if [[ ! -f "${out_pen}" ]]; then
        echo "  [pen ${label}]"
        local out_dir="${PEN_OUT}/_${label}"
        "$PYTHON" scripts/stage_b_generator/measure_penetration.py \
            --input-dir "${EVAL_ROOT}/${input_dir}" \
            --output-dir "${out_dir}" \
            --detail compact
        mv "${out_dir}/summary.json" "${out_pen}"
        rmdir "${out_dir}" 2>/dev/null || true
    else
        echo "  [pen ${label}] cached"
    fi

    if [[ ! -f "${out_jerk}" ]]; then
        echo "  [jerk ${label}]"
        local out_dir="${JERK_OUT}/_${label}"
        "$PYTHON" scripts/stage_b_generator/measure_motion_quality.py \
            --input-dir "${EVAL_ROOT}/${input_dir}" \
            --output-dir "${out_dir}" \
            --detail compact \
            --save-samples
        mv "${out_dir}/summary.json" "${out_jerk}"
        # Move jerk samples to the parent dir for KS distance compute later.
        for f in "${out_dir}"/jerk_samples_*.npz; do
            if [[ -f "$f" ]]; then
                mv "$f" "${JERK_OUT}/${label}_samples.npz"
            fi
        done
        rmdir "${out_dir}" 2>/dev/null || true
    else
        echo "  [jerk ${label}] cached"
    fi
}

echo "=== Unified metric sweep ==="
echo "  EVAL_ROOT=${EVAL_ROOT}"
echo "  PEN_OUT=${PEN_OUT}"
echo "  JERK_OUT=${JERK_OUT}"

while IFS='|' read -r label input_dir; do
    [[ -z "$label" ]] && continue
    run_metric_for_condition "$label" "$input_dir"
done <<< "$CONDITIONS"

echo "=== Done ==="
