#!/usr/bin/env bash
# Pack R38 results for sync-back to laptop.
#
# Intentionally EXCLUDES large transient files:
#   - sampled .npz substitute caches (typically 100+ MB per variant per mode)
#   - model ckpts in runs/training/<vid>/ (~80-200 MB each)
#
# Sync-back contents (always small enough for scp/wsync):
#   - R38 configs (configs/training/stage1p5_r38_*.yaml)
#   - per-variant train logs (runs/round38_matrix/<vid>.log)
#   - per-variant metrics.jsonl (training-time loss curves; small)
#   - downstream-diag summary + stats markdown/JSON
#     (analyses/round38_diag_<vid>/ and _genstage1 mirror)
#   - C41/S4 quality metrics markdown (analyses/round38_quality/)
#   - matrix summary md + log
#
# Input — environment variables (set by run_round38_matrix.sh):
#   ROUND38_STAMP, ROUND38_TARBALL, ROUND38_SUMMARY_MD, ROUND38_SUMMARY_LOG
#   ROUND38_VARIANT_LOG_DIR
#   ROUND38_TRAINED_VIDS, ROUND38_DIAGED_VIDS  (space-separated lists)
#
# Optional opt-ins:
#   ROUND38_PACK_INCLUDE_CKPTS=1     include runs/training/<vid>/final.pt
#   ROUND38_PACK_INCLUDE_NPZ=1       include sampled stage1p5 cache .npz

set -euo pipefail
cd "$(dirname "$0")/../.."

STAMP="${ROUND38_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TARBALL="${ROUND38_TARBALL:-analyses/round38_matrix_results_${STAMP}.tar.gz}"
SUMMARY_MD="${ROUND38_SUMMARY_MD:-analyses/round38_matrix_summary_${STAMP}.md}"
SUMMARY_LOG="${ROUND38_SUMMARY_LOG:-runs/round38_matrix/summary_${STAMP}.log}"
VARIANT_LOG_DIR="${ROUND38_VARIANT_LOG_DIR:-runs/round38_matrix}"
INCLUDE_CKPTS="${ROUND38_PACK_INCLUDE_CKPTS:-0}"
INCLUDE_NPZ="${ROUND38_PACK_INCLUDE_NPZ:-0}"

# Allow either env-list mode (used by the matrix launcher) or explicit
# CLI (--vid stage1p5_r38_a0_full ...). Env wins when set.
TRAINED_VIDS_STR="${ROUND38_TRAINED_VIDS:-}"
DIAGED_VIDS_STR="${ROUND38_DIAGED_VIDS:-}"

# Fall back to "all 4" if no env list was provided — keeps the packer
# usable standalone for re-packing after a separate diag run.
if [[ -z "${TRAINED_VIDS_STR}" && -z "${DIAGED_VIDS_STR}" ]]; then
    DEFAULT_VIDS="stage1p5_r38_b0_baseline stage1p5_r38_b1_init_pose stage1p5_r38_b2_contact_wrist stage1p5_r38_b3_full"
    TRAINED_VIDS_STR="${DEFAULT_VIDS}"
    DIAGED_VIDS_STR="${DEFAULT_VIDS}"
fi

# shellcheck disable=SC2206
TRAINED_VIDS=(${TRAINED_VIDS_STR})
# shellcheck disable=SC2206
DIAGED_VIDS=(${DIAGED_VIDS_STR})

PACK_TARGETS=()

add_if_exists_file() {
    local p="$1"
    [[ -f "${p}" ]] && PACK_TARGETS+=("${p}")
}
add_if_exists_dir() {
    local p="$1"
    [[ -d "${p}" ]] && PACK_TARGETS+=("${p}")
}

# Configs.
for VID in "${TRAINED_VIDS[@]:-}"; do
    [[ -z "${VID}" ]] && continue
    add_if_exists_file "configs/training/${VID}.yaml"
done

# Per-variant training logs + metrics.
for VID in "${TRAINED_VIDS[@]:-}"; do
    [[ -z "${VID}" ]] && continue
    add_if_exists_file "${VARIANT_LOG_DIR}/${VID}.log"
    add_if_exists_file "runs/training/${VID}/metrics.jsonl"
    if [[ "${INCLUDE_CKPTS}" == "1" ]]; then
        add_if_exists_file "runs/training/${VID}/final.pt"
    fi
done

# Downstream diag summaries (oracle + genstage1 modes).
for VID in "${DIAGED_VIDS[@]:-}"; do
    [[ -z "${VID}" ]] && continue
    add_if_exists_dir "analyses/round38_diag_${VID}"
    add_if_exists_dir "analyses/round38_diag_${VID}_genstage1"
done

# Quality metrics directory.
add_if_exists_dir "analyses/round38_quality"

# Matrix-level outputs.
add_if_exists_file "${SUMMARY_MD}"
add_if_exists_file "${SUMMARY_LOG}"

# Sampled .npz caches (opt-in; usually too big to sync).
if [[ "${INCLUDE_NPZ}" == "1" ]]; then
    for VID in "${DIAGED_VIDS[@]:-}"; do
        [[ -z "${VID}" ]] && continue
        add_if_exists_dir "analyses/round32_stage1p5_substitute_conds_r38_${VID}"
        add_if_exists_dir "analyses/round32_stage1p5_substitute_conds_r38_${VID}_genstage1"
    done
fi

if [[ ${#PACK_TARGETS[@]} -eq 0 ]]; then
    echo "[R38 PACK] nothing to pack" >&2
    exit 0
fi

mkdir -p "$(dirname "${TARBALL}")"
echo "[R38 PACK] writing ${TARBALL} (${#PACK_TARGETS[@]} targets)"
tar -czf "${TARBALL}" "${PACK_TARGETS[@]}"
SIZE=$(du -h "${TARBALL}" | cut -f1)
echo "[R38 PACK] wrote ${TARBALL}  (${SIZE})"

# Emit a manifest next to the tarball so the receiver knows what's
# inside without extracting.
MANIFEST="${TARBALL%.tar.gz}_manifest.txt"
{
    echo "# R38 sync-back manifest"
    echo "stamp: ${STAMP}"
    echo "tarball: ${TARBALL}"
    echo "size: ${SIZE}"
    echo "include_ckpts: ${INCLUDE_CKPTS}"
    echo "include_npz: ${INCLUDE_NPZ}"
    echo "trained_vids: ${TRAINED_VIDS[*]:-}"
    echo "diaged_vids: ${DIAGED_VIDS[*]:-}"
    echo
    echo "## Contents:"
    printf '  %s\n' "${PACK_TARGETS[@]}"
} > "${MANIFEST}"
echo "[R38 PACK] manifest at ${MANIFEST}"
