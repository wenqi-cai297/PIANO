#!/usr/bin/env bash
# Pack the small Round-35 Stage-1 OOD diagnostic artifacts that should be
# synced back for local review.
#
# This intentionally excludes sampled substitute-condition caches and alpha
# cache .npz files. For deciding the next experiment we need the audit report,
# downstream summary/stats files, and logs.
#
# Usage:
#   bash scripts/stage_a_generator/pack_round35_stage1_ood_sync.sh
#
# Common overrides:
#   ROUND35_STAGE1P5_VARIANT=stage1p5_r34v2_a_lambda0p005
#   ROUND35_BUCKETS="val"
#   ROUND35_ALPHA_TAGS="alpha000 alpha025 alpha050 alpha075 alpha100"
#   ROUND35_ALLOW_MISSING=1
#   ROUND35_SYNC_OUT=analyses/my_round35_sync.tar.gz

set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --allow-missing) ROUND35_ALLOW_MISSING=1; shift ;;
        -h|--help) sed -n '1,35p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

STAGE1P5_VARIANT="${ROUND35_STAGE1P5_VARIANT:-stage1p5_r34v2_a_lambda0p005}"
BUCKETS_STR="${ROUND35_BUCKETS:-val}"
ALPHA_TAGS_STR="${ROUND35_ALPHA_TAGS:-alpha000 alpha025 alpha050 alpha075 alpha100}"
OUT_ROOT="${ROUND35_OUT_ROOT:-analyses/round35_stage1_coarse_ood_${STAGE1P5_VARIANT}}"
LOG_DIR="${ROUND35_LOG_DIR:-runs/round35_stage1_ood_${STAGE1P5_VARIANT}}"
ALLOW_MISSING="${ROUND35_ALLOW_MISSING:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
TARBALL="${ROUND35_SYNC_OUT:-analyses/round35_stage1_ood_sync_${STAGE1P5_VARIANT}_${STAMP}.tar.gz}"
MANIFEST="${TARBALL%.tar.gz}_manifest.txt"

# shellcheck disable=SC2206
BUCKETS=(${BUCKETS_STR//,/ })
# shellcheck disable=SC2206
ALPHA_TAGS=(${ALPHA_TAGS_STR//,/ })

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

for bucket in "${BUCKETS[@]}"; do
    [[ -z "${bucket}" ]] && continue
    add_required "${OUT_ROOT}/stage1_coarse_ood_audit_${bucket}.md"
    add_required "${OUT_ROOT}/stage1_coarse_ood_audit_${bucket}.json"
done

for tag in "${ALPHA_TAGS[@]}"; do
    [[ -z "${tag}" ]] && continue
    diag_root="analyses/round32_stage1p5_downstream_diag_r35_${STAGE1P5_VARIANT}_${tag}"
    for bucket in "${BUCKETS[@]}"; do
        [[ -z "${bucket}" ]] && continue
        add_required "${diag_root}/sustained_contact_${bucket}/sustained_contact_summary.md"
        add_required "${diag_root}/sustained_contact_${bucket}/sustained_contact_stats.json"
        add_required "${diag_root}/gait_${bucket}/gait_summary.md"
        add_required "${diag_root}/gait_${bucket}/gait_stats.json"
        add_required "${diag_root}/body_action_${bucket}/body_action_summary.md"
        add_required "${diag_root}/body_action_${bucket}/body_action_stats.json"
        add_required "${diag_root}/g1_soft_stance_${bucket}/g1_soft_stance_summary.md"
        add_required "${diag_root}/g1_soft_stance_${bucket}/g1_soft_stance_stats.json"
    done
done

add_optional "${LOG_DIR}/audit_val.log"
add_optional "${LOG_DIR}/audit_train.log"
add_optional "${LOG_DIR}/build_alpha_val.log"
add_optional "${LOG_DIR}/build_alpha_train.log"
for tag in "${ALPHA_TAGS[@]}"; do
    [[ -z "${tag}" ]] && continue
    add_optional "${LOG_DIR}/diag_${tag}.log"
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "[R35 SYNC] missing expected files:"
    printf '  %s\n' "${MISSING[@]}"
    if [[ "${ALLOW_MISSING}" != "1" ]]; then
        echo
        echo "[R35 SYNC] FATAL: set ROUND35_ALLOW_MISSING=1 or pass --allow-missing to pack partial results."
        exit 1
    fi
fi

if [[ ${#PACK_TARGETS[@]} -eq 0 ]]; then
    echo "[R35 SYNC] nothing to pack"
    exit 1
fi

echo "[R35 SYNC] output: ${TARBALL}"
echo "[R35 SYNC] files: $(( ${#PACK_TARGETS[@]} + 1 )) including manifest"
if [[ ${DRY_RUN} -eq 1 ]]; then
    printf '  %s\n' "${PACK_TARGETS[@]}"
    echo "  ${MANIFEST}"
    exit 0
fi

mkdir -p "$(dirname "${TARBALL}")"
{
    echo "Round-35 Stage-1 OOD sync manifest"
    echo "created_at: $(date '+%F %T %z')"
    echo "stage1p5_variant: ${STAGE1P5_VARIANT}"
    echo "buckets: ${BUCKETS[*]}"
    echo "alpha_tags: ${ALPHA_TAGS[*]}"
    echo "out_root: ${OUT_ROOT}"
    echo "log_dir: ${LOG_DIR}"
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
echo "[R35 SYNC] wrote ${TARBALL} (${SIZE})"
echo "[R35 SYNC] manifest: ${MANIFEST}"
