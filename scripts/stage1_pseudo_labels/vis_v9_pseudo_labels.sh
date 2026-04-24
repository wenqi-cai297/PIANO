#!/usr/bin/env bash
# Render a targeted set of pseudo-label visualisations to verify the v9
# extraction (kinematic coupling + foot-joint swap + hand threshold 0.12
# + sitting-normal 0.5).
#
# The sampling is deliberate — not random. Each group tests ONE aspect of
# v9, so a reviewer can answer specific questions about whether v9 did
# what it was supposed to. See ``analyses/2026-04-24_v9_kin_coupling.md``
# for the rationale behind each group.
#
#   1. neuraldome_wrapgrip_recovery     — kin coupling rescues wrap-grip
#   2. omomo_kick_recovery              — foot-joint swap catches kicks
#   3. omomo_scoot_recovery             — sustained foot contact on scoots
#   4. chairs_sit_preservation          — regression guard (sit still works)
#   5. chairs_regression_check          — 3 new v9 zero_contact drops
#   6. neuraldome_bigsofa_sit_failing   — still-failing sofa sit diagnostics
#
# Usage (on server):
#   bash scripts/stage1_pseudo_labels/vis_v9_pseudo_labels.sh                    # all 6 groups
#   bash scripts/stage1_pseudo_labels/vis_v9_pseudo_labels.sh 1 3                # groups 1 + 3 only
#   PIANO_ROOT=/other/path bash scripts/stage1_pseudo_labels/vis_v9_pseudo_labels.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

PIANO_ROOT="${PIANO_ROOT:-/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano}"
OUT="${OUT:-runs/visualizations/v9}"

# If args are given, only run those numbered groups. Otherwise all.
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
    local label="$1"
    local subset="$2"
    shift 2
    local seq_ids=("$@")
    echo "=========================================================="
    echo "[$label] subset=$subset  seqs=${seq_ids[*]}"
    echo "=========================================================="
    piano-visualize-pseudo-labels \
        --data-dir "$PIANO_ROOT/$subset" \
        --pseudo-label-dir "$PIANO_ROOT/$subset/pseudo_labels" \
        --output-dir "$OUT/$label" \
        --seq-ids "${seq_ids[@]}"
}

# --- 1. neuraldome wrap-grip recoveries (kin coupling) ---
# v8 dropped these as zero_contact; v9 should light up hand contact during
# the carry phase as the box translates with the person.
if wants 1; then
    run_group "neuraldome_wrapgrip_recovery" "neuraldome" \
        subject01_box_0 \
        subject01_box_1565 \
        subject01_box_175
fi

# --- 2. omomo kick recoveries (foot-joint swap) ---
# Foot sphere should flash red at the moment of impact + phase flips
# to manipulation as the object accelerates. If contact is only 1-2 frames
# and then gets filtered by min_contact_duration=3, that explains the 12
# residual omomo drops.
if wants 2; then
    run_group "omomo_kick_recovery" "omomo_correct_v2" \
        sub10_floorlamp_021 \
        sub10_largebox_041 \
        sub10_smallbox_041
fi

# --- 3. omomo scoot recoveries (sustained foot contact) ---
# Foot stays in contact throughout the scoot stroke. Kin coupling helps
# even when the foot is outside the 0.06 distance threshold.
if wants 3; then
    run_group "omomo_scoot_recovery" "omomo_correct_v2" \
        sub10_whitechair_051 \
        sub10_whitechair_052
fi

# --- 4. chairs sit preservation (regression guard) ---
# Should look identical to v8. Pelvis red during sit, phase cycles
# approach → stable-contact, support = sitting.
if wants 4; then
    run_group "chairs_sit_preservation" "chairs" \
        Sub0001_Obj116_Seg0_0 \
        Sub0005_Obj116_Seg0_0
fi

# --- 5. chairs regression check (new v9 zero_contact drops) ---
# Diagnoses whether v8's 0.16 hand threshold was carrying these clips
# on borderline "hand hovers near chair" contacts. If vis shows legit
# sit poses with no clear contact, consider a 0.14 compromise on
# chairs-style subsets.
if wants 5; then
    run_group "chairs_regression_check" "chairs" \
        Sub1069_Obj98_Seg0_450 \
        Sub1891_Obj110_Seg0_0_0 \
        Sub1909_Obj33_Seg0_0
fi

# --- 6. neuraldome bigsofa sit failing (unchanged from v8) ---
# The 0.7 → 0.5 normal threshold didn't recover these — vis should show
# whether pelvis joint sits > 20 cm from cushion (pelvis-distance gate
# blocks) or whether the seat is more than 30 cm below the pelvis
# (cylinder-height gate blocks). Either answer defines v10 scope.
if wants 6; then
    run_group "neuraldome_bigsofa_sit_failing" "neuraldome" \
        subject01_bigsofa_1310 \
        subject02_bigsofa_0
fi

echo ""
echo "Done. Output: $OUT"
echo "Per-clip summary.json lives inside each group dir."
