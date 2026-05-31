#!/usr/bin/env bash
# Pack the small Round-36 dynamics artifacts to sync back for local analysis.
#
# Excludes sampled .npz caches and model checkpoints. Includes configs,
# manifests, temporal dynamics reports, downstream summaries/stats, and logs.

set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --allow-missing) ROUND36_ALLOW_MISSING=1; shift ;;
        -h|--help) sed -n '1,35p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

STAGE1_VID="${ROUND36_STAGE1_VID:-stage1_r36_v8v6_dynacc}"
STAGE1P5_VID="${ROUND36_STAGE1P5_VID:-stage1p5_r36_r34v2_a_c41dyn}"
BUCKETS_STR="${ROUND36_BUCKETS:-val}"
ALLOW_MISSING="${ROUND36_ALLOW_MISSING:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
TARBALL="${ROUND36_SYNC_OUT:-analyses/round36_dynamics_sync_${STAMP}.tar.gz}"
MANIFEST="${TARBALL%.tar.gz}_manifest.txt"

STAGE1_TAG="_r36_${STAGE1_VID}"
STAGE1P5_ORACLE_TAG="_r36_${STAGE1P5_VID}_oracle"
STAGE1P5_GEN_TAG="_r36_${STAGE1P5_VID}_genstage1"

# shellcheck disable=SC2206
BUCKETS=(${BUCKETS_STR})

PACK_TARGETS=()
MISSING=()

add_required() {
    local path="$1"
    if [[ -e "${path}" ]]; then
        PACK_TARGETS+=("${path}")
    else
        MISSING+=("${path}")
    fi
}

add_optional() {
    local path="$1"
    if [[ -e "${path}" ]]; then
        PACK_TARGETS+=("${path}")
    fi
}

add_required "configs/training/${STAGE1_VID}.yaml"
add_required "configs/training/${STAGE1P5_VID}.yaml"
add_required "analyses/round36_dynamics_manifest.json"
add_required "analyses/round36_dynamics_manifest.md"

add_optional "runs/training/${STAGE1_VID}/metrics.jsonl"
add_optional "runs/training/${STAGE1P5_VID}/metrics.jsonl"
add_optional "runs/round36_dynamics_training_diag"
add_optional "runs/round31_stage1_downstream${STAGE1_TAG}"
add_optional "runs/round32_stage1p5_downstream${STAGE1P5_ORACLE_TAG}"
add_optional "runs/round32_stage1p5_downstream${STAGE1P5_GEN_TAG}"

for bucket in "${BUCKETS[@]}"; do
    [[ -z "${bucket}" ]] && continue
    add_required "analyses/round36_temporal_dynamics_diag/${STAGE1_VID}/${bucket}/temporal_dynamics_summary.md"
    add_required "analyses/round36_temporal_dynamics_diag/${STAGE1_VID}/${bucket}/temporal_dynamics_stats.json"
    add_required "analyses/round36_temporal_dynamics_diag/${STAGE1P5_VID}_oracle/${bucket}/temporal_dynamics_summary.md"
    add_required "analyses/round36_temporal_dynamics_diag/${STAGE1P5_VID}_oracle/${bucket}/temporal_dynamics_stats.json"
    add_required "analyses/round36_temporal_dynamics_diag/${STAGE1P5_VID}_genstage1/${bucket}/temporal_dynamics_summary.md"
    add_required "analyses/round36_temporal_dynamics_diag/${STAGE1P5_VID}_genstage1/${bucket}/temporal_dynamics_stats.json"
done

add_downstream_group() {
    local root="$1"
    local bucket="$2"
    add_required "${root}/sustained_contact_${bucket}/sustained_contact_summary.md"
    add_required "${root}/sustained_contact_${bucket}/sustained_contact_stats.json"
    add_required "${root}/gait_${bucket}/gait_summary.md"
    add_required "${root}/gait_${bucket}/gait_stats.json"
    add_required "${root}/body_action_${bucket}/body_action_summary.md"
    add_required "${root}/body_action_${bucket}/body_action_stats.json"
    add_required "${root}/g1_soft_stance_${bucket}/g1_soft_stance_summary.md"
    add_required "${root}/g1_soft_stance_${bucket}/g1_soft_stance_stats.json"
}

for bucket in "${BUCKETS[@]}"; do
    [[ -z "${bucket}" ]] && continue
    add_downstream_group "analyses/round31_stage1_downstream_diag${STAGE1_TAG}" "${bucket}"
    add_downstream_group "analyses/round32_stage1p5_downstream_diag${STAGE1P5_ORACLE_TAG}" "${bucket}"
    add_downstream_group "analyses/round32_stage1p5_downstream_diag${STAGE1P5_GEN_TAG}" "${bucket}"
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "[R36 SYNC] missing expected files:"
    printf '  %s\n' "${MISSING[@]}"
    if [[ "${ALLOW_MISSING}" != "1" ]]; then
        echo
        echo "[R36 SYNC] FATAL: pass --allow-missing or set ROUND36_ALLOW_MISSING=1 for partial results."
        exit 1
    fi
fi

if [[ ${#PACK_TARGETS[@]} -eq 0 ]]; then
    echo "[R36 SYNC] nothing to pack"
    exit 1
fi

echo "[R36 SYNC] output: ${TARBALL}"
echo "[R36 SYNC] files: $(( ${#PACK_TARGETS[@]} + 1 )) including manifest"
if [[ ${DRY_RUN} -eq 1 ]]; then
    printf '  %s\n' "${PACK_TARGETS[@]}"
    echo "  ${MANIFEST}"
    exit 0
fi

mkdir -p "$(dirname "${TARBALL}")"
{
    echo "Round-36 dynamics sync manifest"
    echo "created_at: $(date '+%F %T %z')"
    echo "stage1_variant: ${STAGE1_VID}"
    echo "stage1p5_variant: ${STAGE1P5_VID}"
    echo "buckets: ${BUCKETS[*]}"
    echo
    echo "Included files:"
    printf '%s\n' "${PACK_TARGETS[@]}"
    if [[ ${#MISSING[@]} -gt 0 ]]; then
        echo
        echo "Missing expected files:"
        printf '%s\n' "${MISSING[@]}"
    fi
} > "${MANIFEST}"
PACK_TARGETS+=("${MANIFEST}")

tar -czf "${TARBALL}" "${PACK_TARGETS[@]}"
SIZE="$(du -h "${TARBALL}" | cut -f1)"
echo "[R36 SYNC] wrote ${TARBALL} (${SIZE})"
echo "[R36 SYNC] manifest: ${MANIFEST}"
