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
# One group failing (missing JSON, empty pool, vis crash on a bad clip, ...)
# does NOT abort the rest — each group's status is captured and a summary
# is printed at the end. sampler stderr is surfaced verbatim so Python
# tracebacks are visible.
#
# Usage (on server):
#   bash scripts/stage1_pseudo_labels/vis_v9_pseudo_labels.sh            # all 6 groups, 12 each
#   bash scripts/stage1_pseudo_labels/vis_v9_pseudo_labels.sh 1 3        # groups 1 + 3 only
#   bash scripts/stage1_pseudo_labels/vis_v9_pseudo_labels.sh --diagnose # pool counts only, no vis
#   N_PER_GROUP=20 bash scripts/stage1_pseudo_labels/vis_v9_pseudo_labels.sh
#   SEED=7 bash scripts/stage1_pseudo_labels/vis_v9_pseudo_labels.sh
#   PIANO_ROOT=/other/path bash scripts/stage1_pseudo_labels/vis_v9_pseudo_labels.sh

# Intentionally NOT `set -e` — we want one group's failure to be reported,
# not to abort the whole run. Keep `-u` and `-o pipefail` for stricter
# detection of the lesser errors (unset vars, broken pipes).
set -uo pipefail

cd "$(dirname "$0")/../.."

PIANO_ROOT="${PIANO_ROOT:-/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano}"
OUT="${OUT:-runs/visualizations/v9}"
N_PER_GROUP="${N_PER_GROUP:-12}"
SEED="${SEED:-42}"
SAMPLER="scripts/stage1_pseudo_labels/sample_v9_vis_seqs.py"

DIAGNOSE=0
WANT_GROUPS=()
for arg in "$@"; do
    case "$arg" in
        --diagnose) DIAGNOSE=1 ;;
        *) WANT_GROUPS+=("$arg") ;;
    esac
done
if [[ ${#WANT_GROUPS[@]} -eq 0 ]]; then
    WANT_GROUPS=(1 2 3 4 5 6)
fi

wants() {
    local g="$1"
    for x in "${WANT_GROUPS[@]}"; do
        [[ "$x" == "$g" ]] && return 0
    done
    return 1
}

# Status table — filled per group, printed at the end.
declare -A STATUS

call_sampler() {
    # Run sampler, return 0 on success with stdout captured into $REPLY.
    # Captures stderr too and prints it to our stderr unchanged, so Python
    # tracebacks stay visible.
    local out
    if out="$(python "$@" 2> >(tee /dev/stderr))"; then
        REPLY="$out"
        return 0
    else
        REPLY=""
        return 1
    fi
}

run_group() {
    local group_num="$1"
    local label subset seq_ids count

    if ! call_sampler "$SAMPLER" --piano-root "$PIANO_ROOT" --group "$group_num" --emit label; then
        echo "[group $group_num] sampler --emit label failed"
        STATUS[$group_num]="SAMPLER_FAIL(label)"
        return 1
    fi
    label="$REPLY"

    if ! call_sampler "$SAMPLER" --piano-root "$PIANO_ROOT" --group "$group_num" --emit subset; then
        echo "[group $group_num / $label] sampler --emit subset failed"
        STATUS[$group_num]="SAMPLER_FAIL(subset)"
        return 1
    fi
    subset="$REPLY"

    if ! call_sampler "$SAMPLER" --piano-root "$PIANO_ROOT" --group "$group_num" \
        --n "$N_PER_GROUP" --seed "$SEED"; then
        echo "[group $group_num / $label] sampler failed — see Python traceback above"
        STATUS[$group_num]="SAMPLER_FAIL($label)"
        return 1
    fi
    seq_ids="$REPLY"

    count="$(echo "$seq_ids" | wc -w)"
    if [[ -z "$seq_ids" || "$count" -eq 0 ]]; then
        echo "[group $group_num / $label] sampler returned 0 clips — pool is empty"
        echo "    possible causes:"
        echo "      (a) $PIANO_ROOT/$subset/metadata_clean.json missing or not re-generated after latest extraction"
        echo "      (b) $PIANO_ROOT/$subset/cleaning_report.json missing (needed for drop-based groups 5/6)"
        echo "      (c) filter rejected everything — unlikely given local pool size checks"
        STATUS[$group_num]="EMPTY($label)"
        return 1
    fi

    if [[ "$DIAGNOSE" -eq 1 ]]; then
        echo "[group $group_num] $label  —  $count clips from subset=$subset"
        echo "    sample: $(echo "$seq_ids" | cut -d' ' -f1-3) ..."
        STATUS[$group_num]="DIAGNOSE_OK($label, n=$count)"
        return 0
    fi

    echo "=========================================================="
    echo "[group $group_num] $label  —  $count clips from subset=$subset"
    echo "=========================================================="

    # shellcheck disable=SC2086  # $seq_ids must split on whitespace
    if piano-visualize-pseudo-labels \
        --data-dir "$PIANO_ROOT/$subset" \
        --pseudo-label-dir "$PIANO_ROOT/$subset/pseudo_labels" \
        --output-dir "$OUT/$label" \
        --seq-ids $seq_ids; then
        STATUS[$group_num]="OK($label, n=$count)"
    else
        echo "[group $group_num / $label] piano-visualize-pseudo-labels failed"
        STATUS[$group_num]="VIS_FAIL($label)"
        return 1
    fi
}

for g in 1 2 3 4 5 6; do
    if wants "$g"; then
        run_group "$g" || true      # never abort; continue to next group
    fi
done

echo ""
echo "=========================================================="
echo "Summary"
echo "=========================================================="
for g in 1 2 3 4 5 6; do
    if wants "$g"; then
        s="${STATUS[$g]:-NOT_RUN}"
        echo "  group $g: $s"
    fi
done
echo ""
echo "Output: $OUT"
