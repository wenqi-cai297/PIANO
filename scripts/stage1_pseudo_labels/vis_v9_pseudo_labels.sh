#!/usr/bin/env bash
# Render targeted pseudo-label visualisations to verify the v9 extraction
# (kinematic coupling + foot-joint swap + hand threshold 0.12 +
# sitting-normal 0.5).
#
# Each of 6 groups tests ONE aspect of v9. seq-ids are sampled LIVE from
# ``metadata_clean.json`` + ``cleaning_report.json`` by
# ``sample_v9_vis_seqs.py`` — not hard-coded — so the selection stays
# aligned with whatever extraction + cleaning output is on disk, and
# ``N_PER_GROUP`` is easy to tune.
#
#   1. neuraldome_wrapgrip_recovery     — kin coupling rescues wrap-grip
#   2. omomo_kick_recovery              — foot-joint swap catches kicks
#   3. omomo_scoot_recovery             — sustained foot contact on scoots
#   4. chairs_sit_preservation          — regression guard (sit still works)
#   5. chairs_regression_check          — new v9 zero_contact drops
#   6. neuraldome_bigsofa_sit_failing   — still-failing sofa sit diagnostics
#
# Usage (on server):
#   bash scripts/stage1_pseudo_labels/vis_v9_pseudo_labels.sh            # all 6 groups, 12 each
#   bash scripts/stage1_pseudo_labels/vis_v9_pseudo_labels.sh 1 3        # groups 1 + 3 only
#   N_PER_GROUP=20 bash scripts/stage1_pseudo_labels/vis_v9_pseudo_labels.sh
#   SEED=7 bash scripts/stage1_pseudo_labels/vis_v9_pseudo_labels.sh
#   PIANO_ROOT=/other/path bash scripts/stage1_pseudo_labels/vis_v9_pseudo_labels.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

PIANO_ROOT="${PIANO_ROOT:-/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano}"
OUT="${OUT:-runs/visualizations/v9}"
N_PER_GROUP="${N_PER_GROUP:-12}"
SEED="${SEED:-42}"
SAMPLER="scripts/stage1_pseudo_labels/sample_v9_vis_seqs.py"

# Group filter (default = all)
if [[ $# -gt 0 ]]; then
    GROUPS=("$@")
else
    GROUPS=(1 2 3 4 5 6)
fi

wants() {
    local g="$1"
    for x in "${GROUPS[@]}"; do
        [[ "$x" == "$g" ]] && return 0
    done
    return 1
}

run_group() {
    local group_num="$1"
    local label subset seq_ids count
    label="$(python "$SAMPLER" --piano-root "$PIANO_ROOT" --group "$group_num" --emit label)"
    subset="$(python "$SAMPLER" --piano-root "$PIANO_ROOT" --group "$group_num" --emit subset)"
    seq_ids="$(python "$SAMPLER" --piano-root "$PIANO_ROOT" --group "$group_num" \
        --n "$N_PER_GROUP" --seed "$SEED")"

    if [[ -z "$seq_ids" ]]; then
        echo "[group $group_num / $label] no seq_ids sampled — skipping."
        return
    fi

    count="$(echo "$seq_ids" | wc -w)"
    echo "=========================================================="
    echo "[group $group_num] $label  —  $count clips from subset=$subset"
    echo "=========================================================="

    # shellcheck disable=SC2086  # $seq_ids must split on whitespace
    piano-visualize-pseudo-labels \
        --data-dir "$PIANO_ROOT/$subset" \
        --pseudo-label-dir "$PIANO_ROOT/$subset/pseudo_labels" \
        --output-dir "$OUT/$label" \
        --seq-ids $seq_ids
}

for g in 1 2 3 4 5 6; do
    if wants "$g"; then
        run_group "$g"
    fi
done

echo ""
echo "Done. Output: $OUT"
echo "Per-clip summary.json lives inside each group dir."
